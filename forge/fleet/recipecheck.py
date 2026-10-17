"""Prove a machine runs a recipe: start NVIDIA's published server, wait for health, send one chat request,
stop. Runs ON the target inside a GPU grant (the executor wraps it in forge.fleet.gpujob):

    python -m forge.fleet.recipecheck --recipe nano-9b-v2:c1 --gpu-class rtx5090_32g

Prints `RECIPE_CHECK {json}`: ok, startup_s, completion_tokens, wall_s, error. Server log in
$FORGE_ROOT/state/recipecheck/<recipe>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.environ.get("FORGE_ROOT", os.path.expanduser("~/forge"))


def post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--gpu-class")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--startup-timeout", type=int, default=1500)
    a = ap.parse_args(argv)
    sys.path.insert(0, ROOT)
    from forge.recipes import get
    r = get(a.recipe, a.gpu_class)
    port = int(os.environ.get("FORGE_PORT_LO") or 18000)
    cwd = os.path.join(ROOT, "state", "recipe_files", r["base_id"])
    os.makedirs(cwd, exist_ok=True)
    logd = os.path.join(ROOT, "state", "recipecheck")
    os.makedirs(logd, exist_ok=True)
    logp = os.path.join(logd, f"{r['id'].replace(':', '-')}.log")
    vllm = os.path.join(os.path.dirname(a.python), "vllm")
    argv_ = [vllm if os.path.exists(vllm) else "vllm", "serve", r["model"], *r["serve"]["args"],
             "--host", "127.0.0.1", "--port", str(port)]
    env = {**os.environ, **{k: str(v) for k, v in (r["serve"].get("env") or {}).items()}}
    out = {"recipe": r["id"], "model": r["model"], "ok": False, "port": port, "log": logp}
    t0 = time.time()
    with open(logp, "wb") as lf:
        proc = subprocess.Popen(argv_, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while time.time() - t0 < a.startup_timeout:
            if proc.poll() is not None:
                out["error"] = f"server exited rc={proc.returncode}: " + open(logp, errors="replace").read()[-1500:]
                break
            if healthy(port):
                out["startup_s"] = round(time.time() - t0, 1)
                break
            time.sleep(3)
        else:
            out["error"] = "server did not become healthy in time"
        if "startup_s" in out:
            chat = r["workload"].get("chat") or {}
            msgs = [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]
            if chat.get("system_prompt"):
                msgs.insert(0, {"role": "system", "content": chat["system_prompt"]})
            body = {"model": r["serve"].get("served_model_name") or r["model"], "messages": msgs, "max_tokens": 128,
                    "temperature": 0}
            if chat.get("chat_template_kwargs"):
                body["chat_template_kwargs"] = chat["chat_template_kwargs"]
            t1 = time.time()
            res = post(f"http://127.0.0.1:{port}/v1/chat/completions", body)
            out["wall_s"] = round(time.time() - t1, 2)
            out["completion_tokens"] = (res.get("usage") or {}).get("completion_tokens")
            out["text"] = (res["choices"][0]["message"].get("content") or "")[:200]
            out["ok"] = bool(out["completion_tokens"])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
    print("RECIPE_CHECK " + json.dumps(out), flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
