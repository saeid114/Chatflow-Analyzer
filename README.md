# ChatFlow Analyzer 🔍

**An intelligent chatbot transcript analysis toolkit for identifying conversation drop-offs, failed intents, response quality issues, and actionable improvement opportunities.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

ChatFlow Analyzer is a data-driven tool designed for **Conversation Designers** and **Chatbot QA teams** who need to systematically analyze chatbot performance from real conversation transcripts. It parses raw chat logs, detects failure patterns, measures conversation health metrics, and generates prioritized improvement recommendations.

### Key Features

- **Transcript Parsing**: Ingest multi-format chat logs (JSON, CSV) into a standardized conversation model
- **Drop-off Detection**: Identify where and why users abandon conversations
- **Intent Failure Analysis**: Detect fallback triggers, low-confidence responses, and misrouted intents
- **Sentiment Drift Tracking**: Monitor how user sentiment changes throughout a conversation
- **Response Quality Scoring**: Evaluate bot responses for relevance, clarity, and helpfulness
- **Gap Report Generation**: Produce actionable reports with prioritized recommendations
- **Visual Dashboards**: Auto-generate performance charts and funnel visualizations

## Architecture

```
transcripts (JSON/CSV)
        │
        ▼
┌─────────────────┐
│ Transcript Parser│
└────────┬────────┘
         │
    ┌────▼────┐
    │ Analyzer │──► Drop-off Detector
    │  Engine  │──► Intent Failure Analyzer
    │          │──► Sentiment Tracker
    │          │──► Response Quality Scorer
    └────┬────┘
         │
    ┌────▼────────┐
    │ Report Engine│──► Gap Report (Markdown/HTML)
    │              │──► Dashboard Charts (PNG)
    │              │──► Metrics CSV
    └──────────────┘
```

## Installation

```bash
git clone https://github.com/yourusername/chatflow-analyzer.git
cd chatflow-analyzer
pip install -r requirements.txt
```

## Quick Start

```bash
# Analyze sample e-commerce transcripts
python analyzer.py --input data/sample_transcripts.json --output outputs/

# Generate visual dashboard
python dashboard.py --input outputs/analysis_results.json --output outputs/
```

## Sample Output

### Conversation Funnel
The tool identifies where users drop off in the conversation flow:

```
Greeting          ████████████████████████████  100% (500 convos)
Intent Captured   ████████████████████████      87%  (435)
Info Provided     ██████████████████            64%  (320)
Resolution        ████████████████              58%  (290)
Satisfaction      ████████████                  41%  (205)
```

### Gap Report Summary
```
🔴 HIGH PRIORITY
  - "refund_status" intent: 34% fallback rate (152 occurrences)
  - Avg 3.2 re-prompts before resolution on shipping queries
  
🟡 MEDIUM PRIORITY  
  - Sentiment drops >40% after "transfer to agent" message
  - Product recommendation flow has 28% early exit rate

🟢 LOW PRIORITY
  - Greeting variations cause 5% misroute to FAQ
```

## Configuration

Edit `config.yaml` to customize analysis:

```yaml
analysis:
  sentiment_model: "vader"          # vader | textblob
  drop_off_threshold: 2             # messages before considered dropped
  confidence_threshold: 0.65        # below = low confidence flag
  max_reprompt_tolerance: 3         # re-prompts before flagging

reporting:
  format: "markdown"                # markdown | html
  include_charts: true
  top_issues: 10
```

## Project Structure

```
chatflow-analyzer/
├── analyzer.py              # Main analysis engine
├── dashboard.py             # Visualization dashboard generator
├── transcript_parser.py     # Multi-format transcript ingestion
├── config.yaml              # Analysis configuration
├── requirements.txt
├── data/
│   └── sample_transcripts.json
└── outputs/
    ├── analysis_results.json
    ├── gap_report.md
    └── dashboard_charts/
```

## Metrics Computed

| Metric | Description |
|--------|-------------|
| Completion Rate | % of conversations reaching resolution |
| Fallback Rate | % of turns triggering fallback/default |
| Avg Turns to Resolution | Mean messages to resolve user query |
| Sentiment Delta | Change in user sentiment across conversation |
| Re-prompt Rate | How often bot asks user to rephrase |
| Intent Confusion Score | Frequency of intent misroutes |
| CSAT Proxy | Estimated satisfaction from conversation signals |

## Use Cases

- **Pre-launch QA**: Validate conversation flows before deployment
- **Monthly Performance Review**: Track chatbot health metrics over time
- **Competitor Benchmarking**: Compare transcript quality across platforms
- **Prompt Refinement**: Identify which prompts cause confusion or drop-off
- **Localization QA**: Evaluate translated bot performance per region

## License

MIT License - see [LICENSE](LICENSE) for details.
