"""Focused source-adapter tests for Agent-selected market context."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import httpx
import pytest

from tragent.research.agent import retrieval
from tragent.research.agent.retrieval import MarketContextRetriever


def retriever() -> MarketContextRetriever:
    settings = SimpleNamespace(sec_user_agent="Tragent tester test@example.com")
    return MarketContextRetriever(pool=object(), settings=settings)  # type: ignore[arg-type]


def test_yahoo_finance_reads_symbol_rss_before_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_url = ""
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://feeds.finance.yahoo.com"),
        text="""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
          <item>
            <title>Apple announces a product update</title>
            <description>The company described its latest product changes.</description>
            <link>https://example.com/apple-news</link>
            <pubDate>Mon, 15 Jan 2024 12:00:00 +0000</pubDate>
          </item>
          <item>
            <title>A later story</title>
            <description>Published after the study date.</description>
            <link>https://example.com/later-news</link>
            <pubDate>Mon, 22 Jan 2024 12:00:00 +0000</pubDate>
          </item>
        </channel></rss>""",
    )

    def fake_get(url: str, **_kwargs: object) -> httpx.Response:
        nonlocal requested_url
        requested_url = url
        return response

    monkeypatch.setattr(retrieval.httpx, "get", fake_get)

    documents = retriever()._yahoo_news_documents("AAPL", date(2024, 1, 20))

    assert "s=AAPL" in requested_url
    assert len(documents) == 1
    assert documents[0].source_type == "news"
    assert documents[0].source == "Yahoo Finance"
    assert documents[0].published_at == "2024-01-15"
    assert documents[0].excerpt == "The company described its latest product changes."


def test_stocktwits_reads_symbol_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://api.stocktwits.com"),
        json={
            "messages": [
                {
                    "id": 123,
                    "body": "$AAPL momentum looks strong",
                    "created_at": "2024-01-18T12:00:00Z",
                    "user": {"username": "researcher"},
                }
            ]
        },
    )
    monkeypatch.setattr(retrieval.httpx, "get", lambda *_args, **_kwargs: response)

    documents = retriever()._stocktwits_documents("AAPL", date(2024, 1, 20))

    assert len(documents) == 1
    assert documents[0].source_type == "community"
    assert documents[0].source == "StockTwits"
    assert documents[0].url == "https://stocktwits.com/researcher/message/123"


def test_reddit_searches_all_communities_by_the_agents_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_url = ""
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://www.reddit.com"),
        text="""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>AAPL discussion</title>
            <updated>2024-01-18T12:00:00Z</updated>
            <link href="https://www.reddit.com/r/stocks/comments/example" />
            <category term="stocks" label="r/stocks" />
            <content type="html">AAPL community view</content>
          </entry>
        </feed>""",
    )

    def fake_get(url: str, **_kwargs: object) -> httpx.Response:
        nonlocal requested_url
        requested_url = url
        return response

    monkeypatch.setattr(retrieval.httpx, "get", fake_get)

    documents = retriever()._collect(
        symbol="AAPL",
        as_of=date(2024, 1, 20),
        query='"Apple Inc"',
        source="reddit",
    )

    assert "/r/wallstreetbets+stocks+investing/search.rss?q=%22Apple%20Inc%22&" in requested_url
    assert len(documents) == 1
    assert documents[0].source == "Reddit r/stocks"


@pytest.mark.parametrize(
    ("source", "cutoff"),
    [("sec", date(2024, 1, 20)), ("yahoo_finance", date.today())],
)
def test_search_batches_new_documents_and_ranks_only_their_period(
    monkeypatch: pytest.MonkeyPatch,
    source: retrieval.ContextSource,
    cutoff: date,
) -> None:
    documents = (
        retrieval._ContextDocument(
            source_type="official",
            source="SEC",
            title="First filing excerpt",
            excerpt="First material risk excerpt.",
            published_at="2024-01-10",
            url="https://example.com/filing#first",
        ),
        retrieval._ContextDocument(
            source_type="official",
            source="SEC",
            title="Second filing excerpt",
            excerpt="Second material risk excerpt.",
            published_at="2024-01-11",
            url="https://example.com/filing#second",
        ),
    )

    class FakeStore:
        def __init__(self) -> None:
            self.batch_calls: list[list[object]] = []
            self.search_filter: dict[str, object] = {}

        def get(self, _namespace: tuple[str, ...], _key: str) -> None:
            return None

        def batch(self, operations: list[object]) -> list[None]:
            self.batch_calls.append(list(operations))
            return [None] * len(operations)

        def search(self, *_args: object, filter: dict[str, object], **_kwargs: object) -> list:
            self.search_filter = filter
            return []

    store = FakeStore()
    instance = retriever()
    collected_as_of: list[date] = []

    def fake_collect(*, as_of: date, **_kwargs: object) -> tuple[retrieval._ContextDocument, ...]:
        collected_as_of.append(as_of)
        return documents

    monkeypatch.setattr(instance, "_collect", fake_collect)
    monkeypatch.setattr(retrieval, "build_semantic_store", lambda *_args: store)

    result = instance.search(
        symbol="AAPL",
        as_of=date(2024, 1, 20),
        query="material risks",
        source=source,
    )

    assert result == ()
    assert collected_as_of == [cutoff]
    assert len(store.batch_calls) == 1
    assert len(store.batch_calls[0]) == 2
    assert all(isinstance(operation, retrieval.PutOp) for operation in store.batch_calls[0])
    assert store.search_filter["published_at"] == {
        "$gte": "2024-01-10",
        "$lte": cutoff.isoformat(),
    }


class FakeReport:
    def get_item_with_part(self, part: str, item: str) -> str | None:
        # This quarterly report has no Part II risk-factor update.
        return None if (part, item) == ("Part II", "Item 1A") else f"{part} {item} discussion."


class FakeFiling:
    def __init__(self, form: str) -> None:
        self.filing_date = date(2024, 2, 1) if form == "10-K" else date(2024, 5, 1)
        self.filing_url = f"https://www.sec.gov/{form}.htm"

    def obj(self) -> FakeReport:
        return FakeReport()


def test_sec_reads_risk_and_results_sections_of_the_latest_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[object] = []

    class FakeCompany:
        def __init__(self, symbol: str) -> None:
            requests.append(symbol)

        def get_filings(self, *, form: str, filing_date: str, amendments: bool) -> object:
            requests.append((form, filing_date, amendments))
            return SimpleNamespace(latest=lambda: FakeFiling(form))

    monkeypatch.setattr(retrieval, "set_identity", lambda _user_agent: None)
    monkeypatch.setattr(retrieval, "Company", FakeCompany)

    documents = retriever()._sec_documents("AAPL", date(2024, 6, 28))

    assert requests == [
        "AAPL",
        ("10-K", ":2024-06-28", False),
        ("10-Q", ":2024-06-28", False),
    ]
    assert [(document.title, document.excerpt) for document in documents] == [
        ("10-K filed 2024-02-01 · Risk Factors", "Part I Item 1A discussion."),
        ("10-K filed 2024-02-01 · MD&A", "Part II Item 7 discussion."),
        ("10-Q filed 2024-05-01 · MD&A", "Part I Item 2 discussion."),
    ]
    assert {document.url for document in documents} == {
        "https://www.sec.gov/10-K.htm",
        "https://www.sec.gov/10-Q.htm",
    }


def test_sec_skips_reports_a_new_company_has_not_filed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCompany:
        def __init__(self, _symbol: str) -> None:
            pass

        def get_filings(self, *, form: str, **_filters: object) -> object:
            return None if form == "10-K" else SimpleNamespace(latest=lambda: FakeFiling(form))

    monkeypatch.setattr(retrieval, "set_identity", lambda _user_agent: None)
    monkeypatch.setattr(retrieval, "Company", FakeCompany)

    documents = retriever()._sec_documents("NEWCO", date(2024, 6, 28))

    assert [document.title for document in documents] == ["10-Q filed 2024-05-01 · MD&A"]
