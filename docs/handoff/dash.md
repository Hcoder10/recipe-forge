# dash handoff (dash-2, Oct 17 ~12:25)

## Run

TUI, the main UI (stdlib curses, so no install is needed and any python on the rig works):

    ssh -t cube@100.68.107.11 'cd ~/forge && ~/venvs/vllm-bolt/bin/python -m forge.tui'
    ssh -t cube@100.68.107.11 'cd ~/forge && ~/venvs/vllm-bolt/bin/python -m forge.tui --fixture'   # fake demo data
    ssh cube@100.68.107.11 'cd ~/forge && ~/venvs/vllm-bolt/bin/python -m forge.tui --print'        # one text frame

Prompt: `/missions`, `/mission <id>` (focus; bare `/mission` goes back to following the newest), `/leaderboard`,
`/inbox`, `/answer <question_id> <text>`, `/help`, `/quit`. Free text sends an operator note to the focused
mission's Lead. Keys: PgUp/PgDn scroll, Esc returns to the transcript, `?` opens help.

The web dashboard is still running on the rig: http://100.68.107.11:8790 (`?src=fixture` for fake data).
Redeploy both with `bash forge/dash/deploy.sh` from the laptop (tar over ssh). It copies `forge/dash`,
`forge/tui` and `tests/dash`, and restarts only the web process it started (pid and log in `~/forge/state/dash/`).

## Reads (read-only)

Everything in docs/interfaces/dash.md: mission `spec.json`, `events.jsonl` and `row.json`, `gate/ledger.jsonl`,
`brain/facts.jsonl`, `n1/log.jsonl`, the fleet machine and spend files, and `forge/recipes/*.yaml`, under
FORGE_ROOT and FORGE_ROOT/state. The TUI reads `events.jsonl` (transcript) and `inbox.jsonl` (questions)
of the mission it focuses.

## Writes (TUI only, append-only, in the mission dir)

- `answers.jsonl`: `{"question_id", "answer", "t"}`, the format in docs/interfaces/agents.md.
- `operator_notes.jsonl`: `{"t", "by": "operator", "to": "lead", "text"}`. **Not yet agreed with agents**, so the
  mission runner may not read it yet.

## Unfinished

- `/compare <recipe> <gpu>` side-by-side tok/s viewer (web `/compare` too): not started. Plan: connect to two
  given OpenAI-compatible endpoints, stream the same prompt to both, and show live and final tok/s.
- `/gate <id>`, `/facts`, `/n1`, `/pause`, `/stop`: not in the TUI yet. The reader already has the data for
  gate/facts/n1 (`Reader.snapshot()` keys `gates`, `brain`, `n1`). Pause and stop need a write format from agents.
- Tab focus cycling. No layout polish (a single leaderboard strip of 5 non-queued rows).
- Fleet does not push box gate ledgers to `results/boxes/<machine>/gate/ledger.jsonl` yet, so the gate view is
  rig-only.
- Core does not expose per-pair gains in ledger rows yet, so only pair counts show.

## Files

- `forge/dash/reader.py`: the read-only snapshot (all panels), incremental JSONL tails, tolerant field mapping
- `forge/dash/server.py`, `__main__.py`, `static/`: FastAPI + SSE web dashboard
- `forge/dash/fixtures.py`: fake FORGE_ROOT in the real formats, plus a live appender
- `forge/dash/deploy.sh`: deploy to the rig
- `forge/tui/__main__.py`: the curses TUI
- `tests/dash/`: reader, real-process server, and TUI tests (the TUI test needs curses, so it runs on the rig)
- `docs/interfaces/dash.md`: what is read, field by field
