# Thesis Partner

Local web app for a Masters thesis workflow: **binder-style navigation** (linked section pages showing each section’s latest saved draft after analyze), **paste-and-analyze** (APA via **Claude**, **GPTZero** AI scan only on current paste), **on-demand theme fit** (Claude sees all saved section drafts plus thesis memory), thesis memory **paste + chat**, and **research brief**.

## Requirements

- Python 3.11+
- API keys: [Anthropic](https://console.anthropic.com/) and [GPTZero](https://app.gptzero.me/api)

## Configuration

```bash
cp .env.example .env
# Edit .env — keys are gitignored
```

Optional: `CLAUDE_MODEL` (default `claude-sonnet-4-20250514`).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn thesis_partner.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Design system

Generated with [UI UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill): see [design-system/MASTER.md](design-system/MASTER.md) and [design-system/thesis-partner/MASTER.md](design-system/thesis-partner/MASTER.md).

Re-install the Cursor skill (optional):

```bash
npm install -g uipro-cli
uipro init --ai cursor
```

## Data

SQLite lives at `data/thesis_partner.sqlite` (ignored by git). Back it up with your thesis files.

## Limits

Configurable in `thesis_partner/config.py` / env extension later: max characters for analyze, chat, and memory payloads.
