"""Kafka topics between the API and the broker process.

Commands carry only an order ID, since the order itself lives in PostgreSQL. Every topic is keyed
by user, so each user's messages stay in order within one partition. Account snapshots arrive about
once a second and only the latest matters, so that topic keeps an hour; order events keep the
broker's default week.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from pydantic import TypeAdapter

from tragent.config import Settings
from tragent.trading.orders import Order, order_json
from tragent.trading.store import Snapshot

COMMANDS = "trading.order-commands"
ORDER_EVENTS = "trading.order-events"
ACCOUNT_SNAPSHOTS = "trading.account-snapshots"
_PARTITIONS = 3
_SNAPSHOT_RETENTION_MS = 60 * 60 * 1000
_SNAPSHOT_JSON = TypeAdapter(Snapshot)
_DELIVERY_SECONDS = 5


def ensure_topics(settings: Settings) -> None:
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    futures = admin.create_topics(
        [
            NewTopic(COMMANDS, num_partitions=_PARTITIONS, replication_factor=1),
            NewTopic(ORDER_EVENTS, num_partitions=_PARTITIONS, replication_factor=1),
            NewTopic(
                ACCOUNT_SNAPSHOTS,
                num_partitions=_PARTITIONS,
                replication_factor=1,
                config={"retention.ms": str(_SNAPSHOT_RETENTION_MS)},
            ),
        ]
    )
    for future in futures.values():
        try:
            future.result()
        except KafkaException as error:
            if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


def producer(settings: Settings) -> Producer:
    # Idempotence stops the producer's own retries from writing a message twice.
    return Producer(
        {"bootstrap.servers": settings.kafka_bootstrap_servers, "enable.idempotence": True}
    )


def consumer(
    settings: Settings, group_id: str, topics: list[str], *, latest: bool = False
) -> Consumer:
    """A consumer that commits only what its caller has handled."""

    subscriber = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "latest" if latest else "earliest",
            "enable.auto.commit": False,
        }
    )
    subscriber.subscribe(topics)
    return subscriber


def send_command(producer: Producer, order: Order, command: Literal["place", "cancel"]) -> None:
    """Hand a command to the broker process, waiting until Kafka has stored it."""

    producer.produce(
        COMMANDS,
        key=str(order.user_id),
        value=json.dumps({"type": command, "order_id": order.id}),
    )
    if producer.flush(_DELIVERY_SECONDS):
        raise RuntimeError("Kafka did not accept the order command in time")


def publish_event(producer: Producer, order: Order) -> None:
    producer.produce(ORDER_EVENTS, key=str(order.user_id), value=order_json(order))
    producer.poll(0)


def publish_account(producer: Producer, user_id: int, snapshot: Snapshot) -> None:
    value = {
        **_SNAPSHOT_JSON.dump_python(snapshot, mode="json"),
        "synced_at": datetime.now(UTC).isoformat(),
    }
    producer.produce(ACCOUNT_SNAPSHOTS, key=str(user_id), value=json.dumps(value))
    producer.poll(0)


__all__ = [
    "ACCOUNT_SNAPSHOTS",
    "COMMANDS",
    "ORDER_EVENTS",
    "consumer",
    "ensure_topics",
    "producer",
    "publish_account",
    "publish_event",
    "send_command",
]
