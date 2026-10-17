"""TUI logic without curses: event one-liners, notes and answers land in the agreed files."""
import json

import pytest

pytest.importorskip("curses")  # no curses on the Windows laptop; runs on the rig

from forge.dash import fixtures  # noqa: E402
from forge.tui.__main__ import App, fmt_event  # noqa: E402


def test_fmt_event_one_liners():
    role, who, text = fmt_event({"kind": "tool_call", "agent": "w2", "role": "worker", "tool": "bench",
                                 "args": {"config_delta": {"x": 1}}, "result": "+2.1% vs same-GPU ref"})
    assert (role, who) == ("worker", "w2") and text.startswith("bench") and "+2.1%" in text
    assert fmt_event({"kind": "llm_call"}) is None
    assert "REPLAN" in fmt_event({"kind": "decision", "by": "lead", "decision": "replan", "plan": "p"})[2]
    assert fmt_event({"kind": "ask_human", "question_id": "q1", "question": "?"})[0] == "human"


def test_notes_answers_and_views(tmp_path):
    root = str(tmp_path / "fx")
    fixtures.write(root)
    app = App(root)
    app.poll()
    assert app.current() and app.lines
    mid = app.current()
    app.run_command("please check the reference first")
    (note,) = [json.loads(x) for x in open(f"{root}/missions/{mid}/operator_notes.jsonl")]
    assert note["text"] == "please check the reference first" and note["to"] == "lead"
    with open(f"{root}/missions/{mid}/inbox.jsonl", "w") as f:
        f.write(json.dumps({"question_id": "q7", "by": "lead", "question": "use GPU 2?"}) + "\n")
    app.run_command("/inbox")
    assert any("q7" in line for line in app.panel)
    app.run_command("/answer q7 yes, work GPU only")
    (ans,) = [json.loads(x) for x in open(f"{root}/missions/{mid}/answers.jsonl")]
    assert ans == {"question_id": "q7", "answer": "yes, work GPU only", "t": ans["t"]}
    app.run_command("/inbox")
    assert app.panel == ["no open questions"]
    app.run_command("/missions")
    assert len(app.panel) == 10
    app.run_command("/mission m001")
    app.poll()
    assert app.current() == "m001" and any("MISSION END" in x[3] for x in app.lines)
    assert app.run_command("/quit") is False
