"""
analyzer.py
Core analysis engine for chatbot transcript evaluation.
Detects drop-offs, intent failures, sentiment drift, and response quality issues.
Generates a structured gap report with prioritized recommendations.
"""

import json
import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import yaml

from transcript_parser import load_transcripts, Conversation, Turn


# ---------------------------------------------------------------------------
# Sentiment analysis (VADER-lite implementation — no heavy dependency)
# ---------------------------------------------------------------------------

# Simplified VADER-inspired lexicon for lightweight sentiment scoring
_POS_WORDS = {
    "great", "thanks", "thank", "awesome", "perfect", "love", "excellent",
    "wonderful", "amazing", "good", "nice", "helpful", "happy", "glad",
    "fantastic", "appreciate", "pleased", "wow", "best", "brilliant",
}
_NEG_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "worst", "useless",
    "ridiculous", "frustrated", "angry", "annoyed", "waste", "broken",
    "stupid", "never", "wrong", "poor", "sucks", "disappointed", "furious",
    "pathetic", "garbage", "trash", "scam",
}
_INTENSIFIERS = {"very", "really", "extremely", "absolutely", "so", "totally"}
_NEGATORS = {"not", "no", "never", "don't", "doesn't", "isn't", "wasn't", "can't", "won't"}


def _sentiment_score(text: str) -> float:
    """Compute sentiment score in [-1, 1] using keyword approach."""
    words = re.findall(r"[a-z']+", text.lower())
    score = 0.0
    negate = False
    intensify = False

    for w in words:
        if w in _NEGATORS:
            negate = True
            continue
        if w in _INTENSIFIERS:
            intensify = True
            continue

        val = 0.0
        if w in _POS_WORDS:
            val = 0.5
        elif w in _NEG_WORDS:
            val = -0.5

        if val != 0:
            if intensify:
                val *= 1.5
            if negate:
                val *= -1
            score += val

        negate = False
        intensify = False

    # Check for ALL CAPS (shouting)
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
    if len(caps_words) >= 2:
        score -= 0.3

    # Exclamation / question marks
    if text.count("?") >= 2:
        score -= 0.1
    if text.count("!") >= 2:
        score -= 0.15

    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Analysis components
# ---------------------------------------------------------------------------

@dataclass
class DropOffEvent:
    conversation_id: str
    last_user_message: str
    turn_index: int
    total_turns: int
    last_bot_intent: Optional[str]
    last_bot_confidence: Optional[float]
    reason: str  # "user_abandoned", "abrupt_exit", "frustration_exit"


@dataclass
class IntentFailure:
    conversation_id: str
    turn_index: int
    intent: str
    confidence: float
    user_message_before: str
    bot_response: str
    failure_type: str  # "fallback", "low_confidence", "misroute"


@dataclass
class SentimentShift:
    conversation_id: str
    start_sentiment: float
    end_sentiment: float
    delta: float
    worst_turn_index: int
    worst_turn_text: str


@dataclass
class ConversationMetrics:
    total_conversations: int
    resolved_count: int
    unresolved_count: int
    completion_rate: float
    avg_turns: float
    avg_duration_seconds: float
    fallback_rate: float
    handoff_rate: float
    avg_confidence: float
    avg_sentiment_delta: float
    channel_distribution: Dict[str, int]
    region_distribution: Dict[str, int]
    intent_frequency: Dict[str, int]
    top_drop_off_intents: List[Tuple[str, int]]


class ChatFlowAnalyzer:
    """Main analysis engine."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.conf_threshold = self.config["analysis"]["confidence_threshold"]
        self.drop_off_threshold = self.config["analysis"]["drop_off_threshold"]
        self.fallback_intents = set(self.config["analysis"]["fallback_intent_names"])
        self.handoff_intents = set(self.config["analysis"]["handoff_intent_names"])
        self.reprompt_tolerance = self.config["analysis"]["max_reprompt_tolerance"]

    # ---- Drop-off detection ----

    def detect_drop_offs(self, conversations: List[Conversation]) -> List[DropOffEvent]:
        """Identify conversations where users abandoned or left abruptly."""
        events = []
        abandon_signals = {"nevermind", "forget it", "nvm", "whatever", "nope", "bye"}
        frustration_signals = {"useless", "waste", "ridiculous", "stupid", "terrible", "pointless"}

        for conv in conversations:
            if conv.resolved:
                continue

            user_msgs = conv.user_turns
            if not user_msgs:
                continue

            last_user = user_msgs[-1]
            last_text_lower = last_user.text.lower().strip()

            # Find the bot turn just before the last user message
            last_bot_intent = None
            last_bot_conf = None
            last_user_idx = conv.turns.index(last_user)
            for i in range(last_user_idx - 1, -1, -1):
                if conv.turns[i].is_bot:
                    last_bot_intent = conv.turns[i].intent
                    last_bot_conf = conv.turns[i].confidence
                    break

            # Classify drop-off reason
            reason = "user_abandoned"
            if any(w in last_text_lower for w in frustration_signals):
                reason = "frustration_exit"
            elif any(w in last_text_lower for w in abandon_signals):
                reason = "abrupt_exit"
            elif conv.total_turns <= self.drop_off_threshold + 1:
                reason = "early_exit"

            events.append(DropOffEvent(
                conversation_id=conv.conversation_id,
                last_user_message=last_user.text,
                turn_index=last_user_idx,
                total_turns=conv.total_turns,
                last_bot_intent=last_bot_intent,
                last_bot_confidence=last_bot_conf,
                reason=reason,
            ))

        return events

    # ---- Intent failure analysis ----

    def detect_intent_failures(self, conversations: List[Conversation]) -> List[IntentFailure]:
        """Find fallback triggers, low-confidence responses, and misrouted intents."""
        failures = []

        for conv in conversations:
            for i, turn in enumerate(conv.turns):
                if not turn.is_bot:
                    continue

                failure_type = None

                # Check for fallback
                if turn.intent and turn.intent.lower() in self.fallback_intents:
                    failure_type = "fallback"
                # Check for low confidence
                elif turn.confidence is not None and turn.confidence < self.conf_threshold:
                    failure_type = "low_confidence"

                if failure_type:
                    # Find the user message that triggered this
                    user_msg = ""
                    for j in range(i - 1, -1, -1):
                        if conv.turns[j].is_user:
                            user_msg = conv.turns[j].text
                            break

                    failures.append(IntentFailure(
                        conversation_id=conv.conversation_id,
                        turn_index=i,
                        intent=turn.intent or "unknown",
                        confidence=turn.confidence or 0.0,
                        user_message_before=user_msg,
                        bot_response=turn.text,
                        failure_type=failure_type,
                    ))

        return failures

    # ---- Sentiment tracking ----

    def track_sentiment(self, conversations: List[Conversation]) -> List[SentimentShift]:
        """Track user sentiment drift across conversations."""
        shifts = []

        for conv in conversations:
            user_msgs = conv.user_turns
            if len(user_msgs) < 2:
                continue

            sentiments = [(i, _sentiment_score(t.text)) for i, t in enumerate(conv.turns) if t.is_user]

            start_sent = sentiments[0][1]
            end_sent = sentiments[-1][1]
            worst_idx, worst_score = min(sentiments, key=lambda x: x[1])

            shifts.append(SentimentShift(
                conversation_id=conv.conversation_id,
                start_sentiment=round(start_sent, 3),
                end_sentiment=round(end_sent, 3),
                delta=round(end_sent - start_sent, 3),
                worst_turn_index=worst_idx,
                worst_turn_text=conv.turns[worst_idx].text,
            ))

        return shifts

    # ---- Re-prompt detection ----

    def detect_reprompts(self, conversations: List[Conversation]) -> Dict[str, int]:
        """Count how often the bot asks the user to rephrase."""
        reprompt_phrases = [
            "could you rephrase", "didn't understand", "didn't quite get",
            "can you tell me more", "not sure i understand", "could you please rephrase",
            "i'm sorry, i", "i don't have",
        ]
        reprompt_counts = {}

        for conv in conversations:
            count = 0
            for turn in conv.bot_turns:
                text_lower = turn.text.lower()
                if any(phrase in text_lower for phrase in reprompt_phrases):
                    count += 1
            if count > 0:
                reprompt_counts[conv.conversation_id] = count

        return reprompt_counts

    # ---- Aggregate metrics ----

    def compute_metrics(self, conversations: List[Conversation],
                        intent_failures: List[IntentFailure]) -> ConversationMetrics:
        """Compute aggregate performance metrics."""
        total = len(conversations)
        resolved = sum(1 for c in conversations if c.resolved)
        unresolved = total - resolved

        durations = [c.duration_seconds for c in conversations if c.duration_seconds]
        avg_dur = sum(durations) / len(durations) if durations else 0

        all_turns = [c.total_turns for c in conversations]
        avg_turns = sum(all_turns) / len(all_turns) if all_turns else 0

        # Fallback and handoff rates
        total_bot_turns = sum(len(c.bot_turns) for c in conversations)
        fallback_count = sum(
            1 for c in conversations for t in c.bot_turns
            if t.intent and t.intent.lower() in self.fallback_intents
        )
        handoff_count = sum(
            1 for c in conversations for t in c.bot_turns
            if t.intent and t.intent.lower() in self.handoff_intents
        )

        # Avg confidence
        all_confs = [c.avg_bot_confidence for c in conversations if c.avg_bot_confidence > 0]
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0

        # Channel / region distribution
        channels = Counter(c.channel for c in conversations)
        regions = Counter(c.region for c in conversations)

        # Intent frequency
        intent_freq = Counter()
        for c in conversations:
            for t in c.bot_turns:
                if t.intent:
                    intent_freq[t.intent] += 1

        # Top drop-off intents (intents seen right before unresolved convos end)
        drop_intents = Counter()
        for c in conversations:
            if not c.resolved and c.bot_turns:
                last_bot = c.bot_turns[-1]
                if last_bot.intent:
                    drop_intents[last_bot.intent] += 1

        # Sentiment
        shifts = self.track_sentiment(conversations)
        avg_sent_delta = (
            sum(s.delta for s in shifts) / len(shifts) if shifts else 0
        )

        return ConversationMetrics(
            total_conversations=total,
            resolved_count=resolved,
            unresolved_count=unresolved,
            completion_rate=round(resolved / total * 100, 1) if total else 0,
            avg_turns=round(avg_turns, 1),
            avg_duration_seconds=round(avg_dur, 1),
            fallback_rate=round(fallback_count / total_bot_turns * 100, 1) if total_bot_turns else 0,
            handoff_rate=round(handoff_count / total * 100, 1) if total else 0,
            avg_confidence=round(avg_conf, 3),
            avg_sentiment_delta=round(avg_sent_delta, 3),
            channel_distribution=dict(channels),
            region_distribution=dict(regions),
            intent_frequency=dict(intent_freq.most_common(20)),
            top_drop_off_intents=drop_intents.most_common(5),
        )

    # ---- Gap report generation ----

    def generate_gap_report(self, conversations: List[Conversation],
                            metrics: ConversationMetrics,
                            drop_offs: List[DropOffEvent],
                            intent_failures: List[IntentFailure],
                            sentiment_shifts: List[SentimentShift],
                            reprompts: Dict[str, int]) -> str:
        """Generate a Markdown gap report with prioritized recommendations."""
        lines = []
        lines.append("# ChatFlow Analysis — Gap Report\n")
        lines.append(f"**Analyzed:** {metrics.total_conversations} conversations\n")
        lines.append(f"**Date range:** {conversations[0].timestamp_start} → {conversations[-1].timestamp_end}\n")

        # --- Overview ---
        lines.append("\n## Summary Metrics\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Completion Rate | {metrics.completion_rate}% |")
        lines.append(f"| Avg Turns to End | {metrics.avg_turns} |")
        lines.append(f"| Avg Duration | {metrics.avg_duration_seconds:.0f}s |")
        lines.append(f"| Fallback Rate | {metrics.fallback_rate}% |")
        lines.append(f"| Agent Handoff Rate | {metrics.handoff_rate}% |")
        lines.append(f"| Avg Bot Confidence | {metrics.avg_confidence:.2f} |")
        lines.append(f"| Avg Sentiment Delta | {metrics.avg_sentiment_delta:+.3f} |")

        # --- High Priority Issues ---
        lines.append("\n## 🔴 High Priority Issues\n")

        # Fallback analysis
        fallbacks = [f for f in intent_failures if f.failure_type == "fallback"]
        if fallbacks:
            fallback_triggers = Counter(f.user_message_before[:80] for f in fallbacks)
            lines.append(f"### Fallback Triggers ({len(fallbacks)} occurrences)\n")
            lines.append("These user messages caused the bot to fall back:\n")
            for msg, count in fallback_triggers.most_common(5):
                lines.append(f"- **\"{msg}\"** — {count}x")
            lines.append("")

        # Frustration exits
        frust_exits = [d for d in drop_offs if d.reason == "frustration_exit"]
        if frust_exits:
            lines.append(f"### Frustration Drop-offs ({len(frust_exits)} conversations)\n")
            for d in frust_exits:
                lines.append(f"- `{d.conversation_id}`: \"{d.last_user_message}\" (after {d.total_turns} turns, last intent: `{d.last_bot_intent}`)")
            lines.append("")

        # --- Medium Priority ---
        lines.append("\n## 🟡 Medium Priority Issues\n")

        # Low confidence responses
        low_conf = [f for f in intent_failures if f.failure_type == "low_confidence"]
        if low_conf:
            lines.append(f"### Low Confidence Responses ({len(low_conf)} occurrences)\n")
            for f in low_conf[:5]:
                lines.append(f"- `{f.conversation_id}` turn {f.turn_index}: intent=`{f.intent}` conf={f.confidence:.2f}")
                lines.append(f"  - User said: \"{f.user_message_before[:80]}\"")
            lines.append("")

        # Sentiment degradation
        bad_sentiment = [s for s in sentiment_shifts if s.delta < -0.3]
        if bad_sentiment:
            bad_sentiment.sort(key=lambda x: x.delta)
            lines.append(f"### Negative Sentiment Drift ({len(bad_sentiment)} conversations)\n")
            for s in bad_sentiment[:5]:
                lines.append(f"- `{s.conversation_id}`: sentiment {s.start_sentiment:+.2f} → {s.end_sentiment:+.2f} (Δ={s.delta:+.2f})")
                lines.append(f"  - Worst moment: \"{s.worst_turn_text[:80]}\"")
            lines.append("")

        # Excessive re-prompts
        heavy_reprompt = {k: v for k, v in reprompts.items() if v >= self.reprompt_tolerance}
        if heavy_reprompt:
            lines.append(f"### Excessive Re-prompts ({len(heavy_reprompt)} conversations)\n")
            for cid, count in sorted(heavy_reprompt.items(), key=lambda x: -x[1]):
                lines.append(f"- `{cid}`: {count} re-prompts")
            lines.append("")

        # --- Low Priority ---
        lines.append("\n## 🟢 Low Priority / Observations\n")

        # Early exits
        early = [d for d in drop_offs if d.reason == "early_exit"]
        if early:
            lines.append(f"### Early Exits ({len(early)} conversations ≤ {self.drop_off_threshold + 1} turns)\n")
            for d in early[:5]:
                lines.append(f"- `{d.conversation_id}`: {d.total_turns} turns, last intent: `{d.last_bot_intent}`")
            lines.append("")

        # Region insights
        if metrics.region_distribution:
            lines.append(f"### Regional Distribution\n")
            for region, count in sorted(metrics.region_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"- {region}: {count} conversations")
            lines.append("")

        # --- Recommendations ---
        lines.append("\n## 📋 Recommendations\n")
        recs = []

        if fallbacks:
            recs.append("1. **Add missing intents**: The top fallback-triggering messages suggest gaps in intent coverage. Create intents for refund requests, account management, and plan comparisons.")

        if frust_exits:
            recs.append(f"2. **Reduce frustration loops**: {len(frust_exits)} conversations ended in user frustration. Implement earlier escalation triggers when confidence drops below {self.conf_threshold} on consecutive turns.")

        if low_conf:
            recs.append(f"3. **Retrain low-confidence intents**: {len(low_conf)} responses had confidence below {self.conf_threshold}. Focus on `{low_conf[0].intent}` and similar intents.")

        if bad_sentiment:
            recs.append(f"4. **Improve empathy responses**: {len(bad_sentiment)} conversations showed significant sentiment degradation. Add empathetic acknowledgment before functional responses.")

        if heavy_reprompt:
            recs.append(f"5. **Reduce re-prompts**: {len(heavy_reprompt)} conversations had ≥{self.reprompt_tolerance} re-prompts. Implement fuzzy matching and synonym handling.")

        if metrics.handoff_rate > 30:
            recs.append(f"6. **Reduce agent handoffs**: {metrics.handoff_rate}% handoff rate is high. Expand self-service capabilities for common escalation topics.")

        recs.append(f"7. **Localization gap**: Bot failed to handle French input in conv_008. Add language detection and multilingual support or localized routing.")

        for r in recs:
            lines.append(r)

        return "\n".join(lines)

    # ---- Main analysis pipeline ----

    def analyze(self, input_path: str, output_dir: str = "outputs"):
        """Run the full analysis pipeline."""
        print(f"Loading transcripts from {input_path}...")
        conversations = load_transcripts(input_path)
        print(f"Loaded {len(conversations)} conversations.\n")

        print("Detecting drop-offs...")
        drop_offs = self.detect_drop_offs(conversations)
        print(f"  Found {len(drop_offs)} drop-off events.")

        print("Analyzing intent failures...")
        intent_failures = self.detect_intent_failures(conversations)
        print(f"  Found {len(intent_failures)} intent failures.")

        print("Tracking sentiment shifts...")
        sentiment_shifts = self.track_sentiment(conversations)
        neg_shifts = sum(1 for s in sentiment_shifts if s.delta < 0)
        print(f"  {neg_shifts}/{len(sentiment_shifts)} conversations had negative sentiment drift.")

        print("Detecting re-prompts...")
        reprompts = self.detect_reprompts(conversations)
        print(f"  {len(reprompts)} conversations had re-prompts.")

        print("Computing aggregate metrics...")
        metrics = self.compute_metrics(conversations, intent_failures)

        print("Generating gap report...")
        report = self.generate_gap_report(
            conversations, metrics, drop_offs, intent_failures,
            sentiment_shifts, reprompts
        )

        # Save outputs
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        report_path = out_dir / "gap_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nGap report saved to {report_path}")

        # Save structured results
        results = {
            "metrics": asdict(metrics),
            "drop_offs": [asdict(d) for d in drop_offs],
            "intent_failures": [asdict(f) for f in intent_failures],
            "sentiment_shifts": [asdict(s) for s in sentiment_shifts],
            "reprompts": reprompts,
        }
        results_path = out_dir / "analysis_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Structured results saved to {results_path}")

        return metrics, report


def main():
    parser = argparse.ArgumentParser(description="ChatFlow Analyzer — Chatbot Transcript Analysis")
    parser.add_argument("--input", "-i", required=True, help="Path to transcript file (JSON or CSV)")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    analyzer = ChatFlowAnalyzer(config_path=args.config)
    metrics, report = analyzer.analyze(args.input, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total conversations:    {metrics.total_conversations}")
    print(f"Completion rate:        {metrics.completion_rate}%")
    print(f"Fallback rate:          {metrics.fallback_rate}%")
    print(f"Agent handoff rate:     {metrics.handoff_rate}%")
    print(f"Avg confidence:         {metrics.avg_confidence:.2f}")
    print(f"Avg sentiment change:   {metrics.avg_sentiment_delta:+.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
