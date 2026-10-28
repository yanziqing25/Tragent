"""Small external-context adapter backed by LangGraph's semantic Postgres Store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from time import struct_time
from typing import Literal
from urllib.parse import quote

import feedparser
import httpx
from bs4 import BeautifulSoup
from edgar import Company, set_identity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.store.base import PutOp

from tragent.config import Settings
from tragent.db import DatabasePool
from tragent.research.agent.models import ContextSourceType, MarketContextEvidence
from tragent.research.semantic_store import build_semantic_store

type ContextSource = Literal["sec", "yahoo_finance", "stocktwits", "reddit"]

CONTEXT_SOURCES: tuple[ContextSource, ...] = ("sec", "yahoo_finance", "stocktwits", "reddit")

_USER_AGENT = "Tragent/0.1 (+https://gitee.com/yanziqing25/Trading)"
# One search across all three communities: Reddit allows anonymous clients very few requests.
_REDDIT_COMMUNITIES = "wallstreetbets+stocks+investing"
# The company's own account of its risks and results: the latest annual report holds the full
# risk factors, and the latest quarterly report the most recent discussion of results.
_FILING_SECTIONS = {
    "10-K": (("Risk Factors", "Part I", "Item 1A"), ("MD&A", "Part II", "Item 7")),
    "10-Q": (("MD&A", "Part I", "Item 2"), ("Risk Factors", "Part II", "Item 1A")),
}
_SECTION_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)


@dataclass(frozen=True, slots=True)
class _ContextDocument:
    source_type: ContextSourceType
    source: str
    title: str
    excerpt: str
    published_at: str
    url: str


class MarketContextRetriever:
    """Fetch, index, and search bounded context for one confirmed symbol."""

    def __init__(self, pool: DatabasePool, settings: Settings) -> None:
        self.pool = pool
        self.settings = settings

    def search(
        self,
        *,
        symbol: str,
        as_of: date,
        query: str,
        source: ContextSource,
    ) -> tuple[MarketContextEvidence, ...]:
        # News and community feeds carry only the last few days, and context is retrieved after
        # the candidate is frozen, so they run up to today; filings stay within the study period.
        cutoff = as_of if source == "sec" else date.today()
        documents = self._collect(symbol=symbol, as_of=cutoff, query=query, source=source)
        if not documents:
            return ()
        store = build_semantic_store(self.pool, self.settings)
        namespace = ("market_context", symbol)
        pending: list[PutOp] = []
        for document in documents:
            evidence = _evidence(document)
            if store.get(namespace, evidence.id) is None:
                pending.append(
                    PutOp(
                        namespace=namespace,
                        key=evidence.id,
                        value={
                            **evidence.model_dump(mode="json"),
                            "symbol": symbol,
                            "source_key": source,
                            "text": f"{evidence.title}\n{evidence.excerpt}",
                        },
                        index=["text"],
                    )
                )
        if pending:
            store.batch(pending)
        items = store.search(
            namespace,
            query=query,
            filter={
                "source_key": source,
                # Earlier tasks leave older items in the shared store; rank only this fetch's
                # period, never anything after the study date.
                "published_at": {
                    "$gte": min(document.published_at for document in documents),
                    "$lte": cutoff.isoformat(),
                },
            },
            limit=3,
        )
        return tuple(
            MarketContextEvidence.model_validate(
                {name: item.value[name] for name in MarketContextEvidence.model_fields}
            )
            for item in items
        )

    def _collect(
        self,
        *,
        symbol: str,
        as_of: date,
        query: str,
        source: ContextSource,
    ) -> tuple[_ContextDocument, ...]:
        if source == "sec":
            documents = self._sec_documents(symbol, as_of)
        elif source == "yahoo_finance":
            documents = self._yahoo_news_documents(symbol, as_of)
        elif source == "stocktwits":
            documents = self._stocktwits_documents(symbol, as_of)
        else:
            documents = self._reddit_documents(query, as_of)
        return tuple(documents)

    def _sec_documents(self, symbol: str, as_of: date) -> list[_ContextDocument]:
        set_identity(self.settings.sec_user_agent)
        company = Company(symbol)
        documents: list[_ContextDocument] = []
        for form, sections in _FILING_SECTIONS.items():
            filings = company.get_filings(
                form=form, filing_date=f":{as_of.isoformat()}", amendments=False
            )
            if filings is None:  # a company too new to have filed this report by the study end
                continue
            filing = filings.latest()
            report = filing.obj()
            published = filing.filing_date.isoformat()
            for section, part, item in sections:
                documents.extend(
                    _ContextDocument(
                        source_type="official",
                        source="SEC",
                        title=f"{form} filed {published} · {section}",
                        excerpt=chunk,
                        published_at=published,
                        url=filing.filing_url,
                    )
                    for chunk in _SECTION_SPLITTER.split_text(
                        report.get_item_with_part(part, item) or ""
                    )
                )
        return documents

    def _yahoo_news_documents(self, symbol: str, as_of: date) -> list[_ContextDocument]:
        # Yahoo retired the endpoint behind yfinance's Ticker.news; its per-symbol RSS remains.
        documents: list[_ContextDocument] = []
        headers = {"User-Agent": _USER_AGENT}
        url = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline?"
            f"s={quote(symbol)}&region=US&lang=en-US"
        )
        response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        for item in feedparser.parse(response.content).entries:
            published = _feed_date(item.published_parsed)
            if published > as_of.isoformat():
                continue
            title, excerpt, link = item.get("title"), item.get("summary"), item.get("link")
            if not title or not excerpt or not link:
                continue
            documents.append(
                _ContextDocument(
                    source_type="news",
                    source="Yahoo Finance",
                    title=title[:250],
                    excerpt=excerpt[:600],
                    published_at=published,
                    url=link[:500],
                )
            )
        return documents

    def _stocktwits_documents(self, symbol: str, as_of: date) -> list[_ContextDocument]:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{quote(symbol)}.json"
        response = httpx.get(url, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        documents: list[_ContextDocument] = []
        for message in response.json()["messages"][:10]:
            published = str(message["created_at"])[:10]
            body = str(message["body"]).strip()
            if not body or published > as_of.isoformat():
                continue
            username = str(message["user"]["username"])
            message_id = str(message["id"])
            documents.append(
                _ContextDocument(
                    source_type="community",
                    source="StockTwits",
                    title=f"@{username} on ${symbol}",
                    excerpt=body[:600],
                    published_at=published,
                    url=f"https://stocktwits.com/{username}/message/{message_id}",
                )
            )
        return documents

    def _reddit_documents(self, query: str, as_of: date) -> list[_ContextDocument]:
        # Reddit is searched by the Agent's query: a bare ticker such as "U" matches ordinary words.
        documents: list[_ContextDocument] = []
        headers = {"User-Agent": _USER_AGENT}
        url = (
            f"https://www.reddit.com/r/{_REDDIT_COMMUNITIES}/search.rss?"
            f"q={quote(query)}&restrict_sr=on&sort=new&t=week&limit=15"
        )
        response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        response.raise_for_status()
        for entry in feedparser.parse(response.content).entries[:15]:
            title, href = entry.get("title"), entry.get("link")
            published = _feed_date(entry.updated_parsed)
            # Reddit labels each post's community, such as r/stocks, as its category.
            community = entry.tags[0].label if entry.get("tags") else None
            if not title or not href or not community or published > as_of.isoformat():
                continue
            excerpt = _html_text(entry.content[0].value) if entry.get("content") else ""
            documents.append(
                _ContextDocument(
                    source_type="community",
                    source=f"Reddit {community}",
                    title=title[:250],
                    excerpt=excerpt[:600] or title[:600],
                    published_at=published,
                    url=href[:500],
                )
            )
        return documents


def _html_text(value: str) -> str:
    return " ".join(BeautifulSoup(value, "html.parser").stripped_strings)


def _feed_date(parsed: struct_time) -> str:
    """The ISO date of a feed timestamp, which feedparser converts to UTC."""

    return date(*parsed[:3]).isoformat()


def _evidence(document: _ContextDocument) -> MarketContextEvidence:
    identity = f"{document.url}\n{document.title}\n{document.excerpt}"
    evidence_id = sha256(identity.encode("utf-8")).hexdigest()[:24]
    return MarketContextEvidence(
        id=evidence_id,
        source_type=document.source_type,
        source=document.source,
        title=document.title,
        excerpt=document.excerpt,
        published_at=document.published_at,
        url=document.url,
    )


__all__ = ["CONTEXT_SOURCES", "ContextSource", "MarketContextRetriever"]
