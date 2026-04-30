"""
transcript_parser.py
Multi-format transcript ingestion and normalization.
Supports JSON and CSV chat logs, producing standardized Conversation objects.
"""

import json
import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class Turn:
    """A single turn (message) in a conversation."""
    role: str  # 'bot' or 'user'
    text: str
    timestamp: Optional[datetime] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None

    @property
    def is_bot(self) -> bool:
        return self.role == "bot"

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Conversation:
    """A complete conversation session."""
    conversation_id: str
    turns: List[Turn]
    channel: str = "unknown"
    region: str = "unknown"
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    resolved: Optional[bool] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def user_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.is_user]

    @property
    def bot_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.is_bot]

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.timestamp_start and self.timestamp_end:
            return (self.timestamp_end - self.timestamp_start).total_seconds()
        return None

    @property
    def bot_intents(self) -> List[str]:
        return [t.intent for t in self.bot_turns if t.intent]

    @property
    def avg_bot_confidence(self) -> float:
        confs = [t.confidence for t in self.bot_turns if t.confidence is not None]
        return sum(confs) / len(confs) if confs else 0.0


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO format timestamp."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def load_json_transcripts(filepath: str) -> List[Conversation]:
    """Load conversations from a JSON transcript file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    for entry in data:
        turns = []
        for t in entry.get("turns", []):
            turns.append(Turn(
                role=t["role"],
                text=t["text"],
                timestamp=parse_timestamp(t.get("timestamp", "")),
                intent=t.get("intent"),
                confidence=t.get("confidence"),
            ))

        conv = Conversation(
            conversation_id=entry["conversation_id"],
            turns=turns,
            channel=entry.get("channel", "unknown"),
            region=entry.get("region", "unknown"),
            timestamp_start=parse_timestamp(entry.get("timestamp_start", "")),
            timestamp_end=parse_timestamp(entry.get("timestamp_end", "")),
            resolved=entry.get("resolved"),
        )
        conversations.append(conv)

    return conversations


def load_csv_transcripts(filepath: str) -> List[Conversation]:
    """Load conversations from a CSV transcript file.
    Expected columns: conversation_id, role, text, timestamp, intent, confidence
    """
    rows_by_conv = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["conversation_id"]
            if cid not in rows_by_conv:
                rows_by_conv[cid] = []
            rows_by_conv[cid].append(row)

    conversations = []
    for cid, rows in rows_by_conv.items():
        turns = []
        for r in rows:
            conf = r.get("confidence", "")
            turns.append(Turn(
                role=r["role"],
                text=r["text"],
                timestamp=parse_timestamp(r.get("timestamp", "")),
                intent=r.get("intent") or None,
                confidence=float(conf) if conf else None,
            ))

        timestamps = [t.timestamp for t in turns if t.timestamp]
        conv = Conversation(
            conversation_id=cid,
            turns=turns,
            timestamp_start=min(timestamps) if timestamps else None,
            timestamp_end=max(timestamps) if timestamps else None,
        )
        conversations.append(conv)

    return conversations


def load_transcripts(filepath: str) -> List[Conversation]:
    """Auto-detect format and load transcripts."""
    path = Path(filepath)
    if path.suffix.lower() == ".json":
        return load_json_transcripts(filepath)
    elif path.suffix.lower() == ".csv":
        return load_csv_transcripts(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
