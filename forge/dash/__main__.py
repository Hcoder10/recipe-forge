"""python -m forge.dash [--root ~/forge] [--host 0.0.0.0] [--port 8790] [--no-fixture]

Serves the real files under --root at /, and fake demo data at /?src=fixture.
"""
from __future__ import annotations

import argparse
import os
import threading


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m forge.dash")
    ap.add_argument("--root", default=os.environ.get("FORGE_ROOT", os.path.expanduser("~/forge")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between reader polls")
    ap.add_argument("--no-fixture", action="store_true",
                    help="don't also serve fake data at ?src=fixture (written under <root>/dash_fixture)")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    fixture_root = None
    if not args.no_fixture:
        from forge.dash import fixtures
        fixture_root = os.path.join(root, "dash_fixture")
        fixtures.write(fixture_root)
        threading.Thread(target=fixtures.live_loop, args=(fixture_root,), daemon=True).start()

    import uvicorn
    from forge.dash.server import create_app

    uvicorn.run(create_app(root, args.poll, fixture_root), host=args.host, port=args.port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
