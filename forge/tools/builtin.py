"""The agent tool set. High-level tools first; raw sh/py last.

Tool text states capabilities and procedure only (NO-LEAK): never what to try.
"""
from __future__ import annotations

import difflib
import json
import posixpath
import time

from forge.tools.bundle import canon, hash_files, normalize_delta
from forge.tools.registry import EnvBlocked, Registry, Tool, ToolError

LANES = ["config", "env", "runtime", "kernels", "drafter", "system"]
W, L, V = "worker", "lead", "verifier"


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}
O = {"type": "object"}


# ---------------------------------------------------------------- paths
def _resolve(ctx, path: str, for_write: bool = False) -> tuple[str, str, str | None]:
    """-> (kind, abs_path, overlay_rel). kind: overlay (file in the recipe venv's site-packages, edited as a
    copy inside the workspace) | ws (file in the workspace) | ro (readable reference: model config)."""
    m = ctx.mission
    p = (path or "").strip().replace("\\", "/")
    if not p:
        raise ToolError("path is empty")
    if ".." in p.split("/"):
        raise ToolError("path may not contain '..'")
    sp = m.site_packages.rstrip("/")
    if p.startswith("/"):
        if p == sp or p.startswith(sp + "/"):
            rel = p[len(sp) + 1:]
            return "overlay", p, rel
        if p == m.ws.rstrip("/") or p.startswith(m.ws.rstrip("/") + "/"):
            return "ws", p, None
        for ro in m.readable_roots:
            if p == ro.rstrip("/") or p.startswith(ro.rstrip("/") + "/"):
                if for_write:
                    raise ToolError(f"{path} is read-only")
                return "ro", p, None
        raise ToolError("path must be inside the workspace, the recipe venv's site-packages, or the model dir")
    if p.startswith("ws/"):
        return "ws", posixpath.join(m.ws, p[3:]), None
    top = p.split("/", 1)[0]
    if top in m.packages():
        return "overlay", posixpath.join(sp, p), p
    return "ws", posixpath.join(m.ws, p), None


def _overlay_path(ctx, rel: str) -> str:
    return posixpath.join(ctx.mission.overlay_root(ctx.agent), rel)


def _read_current(ctx, path: str) -> tuple[str, str]:
    kind, ap, rel = _resolve(ctx, path)
    ex = ctx.executor
    if kind == "overlay":
        op = _overlay_path(ctx, rel)
        if ex.exists(op):
            return ex.read_text(op), "overlay"
        if not ex.exists(ap):
            raise ToolError(f"no such file: {path}")
        return ex.read_text(ap), "original"
    if not ex.exists(ap):
        raise ToolError(f"no such file: {path}")
    return ex.read_text(ap), kind


# ---------------------------------------------------------------- high-level
def t_profile(ctx):
    m = ctx.mission
    info = {"recipe_id": m.recipe_id, "model": m.recipe["model"], "serve_args": m.recipe["serve"]["args"],
            "serve_env": m.recipe["serve"].get("env") or {}, "workload": m.recipe["workload"],
            "metric": m.metric, "machine": m.machine.get("name"), "venv_python": m.python,
            "site_packages": m.site_packages, "workspace": m.ws, "lanes": LANES}
    r = ctx.executor.run("nvidia-smi --query-gpu=index,name,memory.total,power.limit,clocks.max.sm "
                         "--format=csv,noheader; " + f"{m.python} -c 'import vllm,torch;"
                         "print(\"vllm\",vllm.__version__);print(\"torch\",torch.__version__,torch.version.cuda)'",
                         timeout_s=120, mission_id=m.id, label="profile")
    info["machine_report"] = (r.stdout or r.stderr).strip()[-1500:]
    ref = m.reference()
    info["reference"] = ref or "not measured in this mission yet (bench() with no change measures it)"
    return info


def t_bench(ctx, config_delta=None, env=None, bundle=None, n_prompts=None, fresh_reference=False, label="",
            claim="", predict_pct=None):
    m = ctx.mission
    bundle_dir = None
    if bundle:
        bundle_dir = m.bundle_dir(bundle)
        if not ctx.executor.exists(posixpath.join(bundle_dir, "manifest.json")):
            raise ToolError(f"no bundle {bundle!r}; make_bundle returns the id")
    delta = normalize_delta(config_delta) if config_delta else None
    t0 = time.time()
    out = ctx.services.bench.run(m, config_delta=delta, env=env or None, bundle_dir=bundle_dir,
                                 n_prompts=n_prompts, fresh_reference=fresh_reference,
                                 label=label or ("ref" if not (delta or env or bundle) else "cand"))
    res = {"gpu": out.get("gpu", {}).get("name"), "metric": m.metric,
           "reference": _brief_measure(out.get("reference")), "reference_cached": out.get("reference_cached")}
    if "candidate" in out:
        res["candidate"] = _brief_measure(out["candidate"])
        res["delta_pct"] = out.get("delta_pct")
    res["bench_id"] = m.record_bench(ctx.agent, {"config_delta": delta, "env": env, "bundle": bundle,
                                                 "claim": claim[:300], "predict_pct": predict_pct,
                                                 "result": res, "wall_s": round(time.time() - t0, 1)})
    det = getattr(ctx.services, "detector", None)
    if det and "delta_pct" in res:
        det.observe_prediction(ctx.agent, predict_pct, res["delta_pct"], res["bench_id"])
    res["note"] = "quick bench on visible prompts, same GPU as its reference; only the gate verifies a win"
    return res


def _brief_measure(r):
    if not r:
        return None
    keep = ("value", "metric", "n", "tokens", "decode_tok_s", "output_tok_s", "ttft_ms_mean", "startup_s")
    out = {k: r[k] for k in keep if k in r}
    pp = r.get("per_prompt_decode")
    if pp:
        s = sorted(pp)
        out["per_prompt_min_med_max"] = [s[0], s[len(s) // 2], s[-1]]
    return out


def t_read(ctx, path, start=1, lines=150):
    kind, ap, rel = _resolve(ctx, path)
    target = _overlay_path(ctx, rel) if kind == "overlay" and ctx.executor.exists(_overlay_path(ctx, rel)) else ap
    if _file_run(ctx, ["test", "-d", target], "read", allowed=(0, 1)).rc == 0:
        return t_ls(ctx, path)
    text, src = _read_current(ctx, path)
    all_lines = text.splitlines()
    start = max(1, int(start))
    lines = max(1, min(400, int(lines)))
    chunk = all_lines[start - 1:start - 1 + lines]
    body = "\n".join(f"{start + i}: {l}" for i, l in enumerate(chunk))
    return {"path": path, "source": src, "total_lines": len(all_lines), "from": start,
            "to": start + len(chunk) - 1, "text": body}


def t_search(ctx, pattern, path="vllm", max_hits=60, context=0, include=None, ignore_case=False):
    kind, ap, _ = _resolve(ctx, path)
    includes = [include] if include is not None else ["*." + ext for ext in
               ("py", "cu", "h", "json", "cuh", "cpp", "yaml", "toml", "txt", "md")]
    cmd = ["grep", "-rnE", *["--include=" + glob for glob in includes], "-C", str(max(0, int(context)))]
    if ignore_case:
        cmd.append("-i")
    r = ctx.executor.run(cmd + ["--", pattern, ap], timeout_s=60, mission_id=ctx.mission.id, label="search")
    if r.infra_error:
        raise EnvBlocked(r.infra_error)
    if r.rc not in (0, 1):
        raise ToolError(f"search failed: {(r.stderr or '').strip()[:200]}")
    hits = r.stdout.splitlines()
    sp = ctx.mission.site_packages.rstrip("/") + "/"
    hits = [h.replace(sp, "") for h in hits]
    max_hits = max(1, int(max_hits))
    return {"hits": hits[:max_hits], "total": len(hits), "truncated": len(hits) > max_hits}


def t_edit(ctx, path, old=None, new=None, content=None, replace_all=False):
    kind, ap, rel = _resolve(ctx, path, for_write=True)
    ex = ctx.executor
    target = _overlay_path(ctx, rel) if kind == "overlay" else ap
    before = _read_current(ctx, path)[0] if ex.exists(target) or ex.exists(ap) else ""
    if content is not None:
        if old is not None:
            raise ToolError("give either content (whole file) or old+new (one replacement), not both")
        ex.write_text(target, content)
        return {"path": path, "wrote": len(content), "as": "overlay" if kind == "overlay" else "workspace file",
                "diff": _file_diff(path, before, content)[:2000]}
    if old is None or new is None:
        raise ToolError("edit needs old and new (exact text replacement) or content (whole file)")
    text, _ = _read_current(ctx, path)
    if not old:
        raise ToolError("old text must not be empty")
    n = text.count(old)
    span = None
    if n == 0:
        span = _whitespace_match(text, old)
        n = 1
    if n > 1 and not replace_all:
        raise ToolError(f"old text occurs {n} times; include more surrounding lines so it is unique")
    after = text[:span[0]] + new + text[span[1]:] if span else text.replace(old, new, n if replace_all else 1)
    ex.write_text(target, after)
    return {"path": path, "replaced": n if replace_all else 1,
            "as": "overlay" if kind == "overlay" else "workspace file", "diff": _file_diff(path, text, after)[:2000]}


def _file_diff(path, before, after):
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                        f"a/{path}", f"b/{path}", n=2))


def _whitespace_match(text, old):
    lines = text.splitlines(keepends=True)
    wanted = old.splitlines(keepends=True)
    for normalize in (str.rstrip, str.strip):
        matches = []
        for i in range(len(lines) - len(wanted) + 1):
            chunk = lines[i:i + len(wanted)]
            if list(map(normalize, chunk)) == list(map(normalize, wanted)):
                actual = "".join(chunk)
                if not old.endswith(("\n", "\r")):
                    actual = actual.rstrip("\r\n")
                start = sum(len(line) for line in lines[:i])
                matches.append((start, start + len(actual)))
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ToolError("old text has multiple whitespace-tolerant matches; include more context")
    raise ToolError("old text not found, including whitespace-tolerant matching")


def _file_run(ctx, cmd, label, allowed=(0,)):
    r = ctx.executor.run(cmd, timeout_s=60, mission_id=ctx.mission.id, label=label)
    if r.infra_error:
        raise EnvBlocked(r.infra_error)
    if r.rc not in allowed:
        detail = " ".join((r.stderr or "path missing or invalid").splitlines())[:200]
        raise ToolError(f"{label} failed: {detail}")
    return r


def _listing_roots(ctx, path):
    kind, ap, rel = _resolve(ctx, path)
    roots = [(ap, False)]
    if kind == "overlay":
        op = _overlay_path(ctx, rel)
        if ctx.executor.exists(op):
            roots.append((op, True))
    return roots


def t_ls(ctx, path=".", depth=1):
    entries = {}
    roots = _listing_roots(ctx, path)
    for root, edited in roots:
        if not ctx.executor.exists(root) and len(roots) > 1:
            continue
        _file_run(ctx, ["test", "-d", root], "ls")
        r = _file_run(ctx, ["find", root, "-mindepth", "1", "-maxdepth", str(max(1, int(depth))),
                            "-printf", "%P\\0%y\\0%s\\0"], "ls")
        fields = r.stdout.split("\0")
        for i in range(0, len(fields) - 1, 3):
            name, typ, size = fields[i:i + 3]
            entries[name] = {"path": name, "type": "directory" if typ == "d" else "file",
                             "size": int(size), "edited": edited}
    rows = [entries[name] for name in sorted(entries)]
    return {"path": path, "entries": rows[:300], "truncated": len(rows) > 300}


def t_glob(ctx, pattern, path="vllm"):
    # find's -path treats ** like *, so also include each zero-directory variant.
    patterns = {pattern}
    pending = [pattern]
    while pending:
        current = pending.pop()
        start = 0
        while (at := current.find("**/", start)) >= 0:
            variant = current[:at] + current[at + 3:]
            if variant not in patterns:
                patterns.add(variant)
                pending.append(variant)
            start = at + 3
    files = set()
    roots = _listing_roots(ctx, path)
    for root, _ in roots:
        if not ctx.executor.exists(root) and len(roots) > 1:
            continue
        predicates = []
        for p in sorted(patterns):
            if predicates:
                predicates.append("-o")
            predicates += ["-path", root.rstrip("/") + "/" + p]
        r = _file_run(ctx, ["find", root, "-type", "f", "(", *predicates, ")", "-printf", "%P\\0"], "glob")
        files.update(posixpath.join(path, name) for name in r.stdout.split("\0") if name)
    return {"files": sorted(files)[:200], "truncated": len(files) > 200}


def t_apply_patch(ctx, patch):
    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ToolError("patch must have Begin Patch and End Patch markers")
    # Stage by resolved destination, so aliases and repeated sections see earlier staged changes.
    staged, originals, names = {}, {}, {}

    def target(path):
        kind, ap, rel = _resolve(ctx, path, for_write=True)
        dest = _overlay_path(ctx, rel) if kind == "overlay" else ap
        if dest not in staged:
            value = _read_current(ctx, path)[0] if ctx.executor.exists(dest) or ctx.executor.exists(ap) else None
            staged[dest] = originals[dest] = value
            names[dest] = path
        return dest, kind, ap

    i = 1
    while i < len(lines) - 1:
        header = lines[i]
        operation = next((op for op in ("Add", "Delete", "Update") if header.startswith(f"*** {op} File: ")), None)
        if operation is None:
            raise ToolError(f"invalid patch header: {header!r}")
        path = header.split(": ", 1)[1]
        dest, kind, ap = target(path)
        before = staged[dest]
        i += 1
        if operation == "Delete":
            if before is None:
                raise ToolError(f"no such file: {path}")
            staged[dest] = None
            continue
        if operation == "Add":
            if before is not None:
                raise ToolError(f"file already exists: {path}")
            added = []
            while i < len(lines) - 1 and lines[i].startswith("+"):
                added.append(lines[i][1:])
                i += 1
            staged[dest] = "\n".join(added) + ("\n" if added else "")
            continue
        if before is None:
            raise ToolError(f"no such file: {path}")
        move = None
        if lines[i].startswith("*** Move to: "):
            move = lines[i][13:]
            i += 1
        current = before.splitlines()
        cursor, hunks = 0, 0
        while i < len(lines) - 1 and (lines[i] == "@@" or lines[i].startswith("@@ ")):
            anchor = lines[i][2:].strip()
            i += 1
            body = []
            while i < len(lines) - 1 and lines[i][:1] in (" ", "-", "+"):
                body.append(lines[i])
                i += 1
            eof = lines[i] == "*** End of File"
            if eof:
                i += 1
            if not body:
                raise ToolError(f"{path}: empty hunk")
            old = [line[1:] for line in body if line[0] != "+"]
            new = [line[1:] for line in body if line[0] != "-"]
            start = cursor
            if anchor:
                found = next((j for j in range(cursor, len(current)) if current[j] == anchor), None)
                if found is None:
                    raise ToolError(f"{path}: hunk did not match: {body[0]!r} (anchor {anchor!r})")
                start = found + 1
            match = next((j for j in range(start, len(current) - len(old) + 1)
                          if current[j:j + len(old)] == old and (not eof or j + len(old) == len(current))), None)
            if match is None:
                raise ToolError(f"{path}: hunk did not match: {body[0]!r}")
            current[match:match + len(old)] = new
            cursor = match + len(new)
            hunks += 1
        if not hunks:
            raise ToolError(f"{path}: update needs a hunk")
        after = "\n".join(current) + ("\n" if current and before.endswith("\n") else "")
        if move:
            moved, _, _ = target(move)
            if moved != dest and staged[moved] is not None:
                raise ToolError(f"move destination already exists: {move}")
            staged[dest] = None
            staged[moved] = after
        else:
            staged[dest] = after
    diffs = []
    for dest, value in staged.items():
        # Removing a copy restores the underlying original, rather than deleting it.
        kind, ap, _ = _resolve(ctx, names[dest])
        effective = ctx.executor.read_text(ap) if value is None and kind == "overlay" and ctx.executor.exists(ap) else value
        diffs.append(_file_diff(names[dest], originals[dest] or "", effective or ""))
    for dest, value in staged.items():
        if value is None:
            ctx.executor.remove(dest)
        else:
            ctx.executor.write_text(dest, value)
    return {"files": list(names.values()), "diff": "".join(diffs)[:4000]}


def t_diff(ctx, path=None):
    m = ctx.mission
    ex = ctx.executor
    rels = [path] if path else m.overlay_files(ctx.agent)
    out = []
    for rel in rels:
        kind, ap, r = _resolve(ctx, rel)
        if kind != "overlay":
            continue
        op = _overlay_path(ctx, r)
        if not ex.exists(op):
            continue
        a = ex.read_text(ap).splitlines(keepends=True) if ex.exists(ap) else []
        b = ex.read_text(op).splitlines(keepends=True)
        out.append("".join(difflib.unified_diff(a, b, f"a/{r}", f"b/{r}", n=2)))
    d = "\n".join(x for x in out if x)
    return {"files": len([x for x in out if x]), "diff": d or "(no source changes)"}


def t_revert(ctx, path):
    kind, ap, rel = _resolve(ctx, path)
    if kind != "overlay":
        raise ToolError("revert applies to edited venv files")
    op = _overlay_path(ctx, rel)
    ctx.executor.remove(op)
    return {"path": path, "reverted": True}


def t_make_bundle(ctx, config_delta=None, env=None, include_edits=True, kernels=None, note=""):
    m = ctx.mission
    ex = ctx.executor
    files: dict[str, bytes] = {}
    overlays = {}
    if include_edits:
        for rel in m.overlay_files(ctx.agent):
            text = ex.read_text(_overlay_path(ctx, rel))
            files[f"overlays/{rel}"] = text.encode()
            overlays[rel] = f"overlays/{rel}"
    kfiles = []
    for k in kernels or []:
        kind, ap, _ = _resolve(ctx, k)
        if kind != "ws":
            raise ToolError(f"kernel files must be workspace files: {k}")
        rel = posixpath.relpath(ap, posixpath.join(m.ws, "kernels")) if ap.startswith(
            posixpath.join(m.ws, "kernels") + "/") else posixpath.basename(ap)
        files[f"kernels/{rel}"] = ex.read_text(ap).encode()
        kfiles.append(f"kernels/{rel}")
    delta = normalize_delta(config_delta)
    if not (delta or env or overlays or kfiles):
        raise ToolError("empty bundle: give a config_delta, env, edits or kernels")
    manifest = {"config_delta": delta, "env": env or {}, "overlays": overlays, "kernels": kfiles,
                "drafter": None, "recipe_id": m.recipe_id, "note": note[:500]}
    files["manifest.json"] = canon(manifest).encode()
    h = hash_files(files)
    bid = h[:16]
    bdir = m.bundle_dir(bid)
    if not ex.exists(posixpath.join(bdir, "manifest.json")):
        for rel, data in files.items():
            ex.write_text(posixpath.join(bdir, rel), data.decode())
    m.register_bundle(bid, {"hash": h, "manifest": manifest, "by": ctx.agent})
    return {"bundle": bid, "hash": h, "files": sorted(files), "config_delta": delta, "env": env or {}}


def t_kernel_test(ctx, path, args="", vram_gb=8, timeout_s=600):
    kind, ap, _ = _resolve(ctx, path)
    if kind != "ws":
        raise ToolError("kernel_test runs a workspace script")
    m = ctx.mission
    cmd = f"cd {m.ws} && {m.python} {ap} {args}"
    r = ctx.executor.run(cmd, timeout_s=min(1800, int(timeout_s)), kind="bench", vram_gb=float(vram_gb), gpus=1,
                         mission_id=m.id, label="kernel_test")
    if r.infra_error:
        raise EnvBlocked(f"kernel_test could not run: {r.infra_error}")
    return {"rc": r.rc, "stdout": r.stdout[-3000:], "stderr": r.stderr[-1500:], "elapsed_s": round(r.elapsed_s, 1)}


def t_request_gate(ctx, bundle, claim, bench_id):
    return ctx.services.gate_queue.submit(ctx, bundle=bundle, claim=claim, bench_id=bench_id)


def t_gate_requests(ctx):
    return {"requests": ctx.services.gate_queue.listing(), "bar": ctx.services.gate_queue.bar_text()}


def t_review_gate(ctx, request_id, approve, reason):
    return ctx.services.gate_queue.review(ctx, request_id=request_id, approve=approve, reason=reason)


def t_gate_status(ctx):
    rows = ctx.services.gate.ledger(mission_id=ctx.mission.id)
    return {"gates": rows, "pending_review": [r["id"] for r in ctx.services.gate_queue.listing()
                                              if r["status"] == "pending_review"]}


def t_note(ctx, text):
    return {"note_id": ctx.services.notes.add(ctx.agent, text)}


def t_fact_add(ctx, claim, lane, status, evidence):
    if lane not in LANES:
        raise ToolError(f"lane must be one of {', '.join(LANES)}")
    fid = ctx.services.brain.add({"claim": claim, "lane": lane, "status": status, "evidence": evidence,
                                  "recipe_id": ctx.mission.recipe_id, "gpu_class": ctx.mission.gpu_class,
                                  "mission_id": ctx.mission.id, "by": ctx.agent})
    return {"fact_id": fid}


def t_facts(ctx, lane=None, query=None, limit=30):
    rows = ctx.services.brain.view(ctx.mission.recipe_id, lane=lane, query=query, limit=limit)
    return {"facts": rows}


def t_decide(ctx, decision, trigger_ids, evidence, plan=""):
    return ctx.services.decisions.decide(ctx.agent, decision, trigger_ids, evidence, plan)


def t_ask_human(ctx, question):
    qid = ctx.services.inbox.ask(ctx.agent, question)
    return {"question_id": qid, "note": "queued for a human; keep working, an answer will appear in your context"}


def t_assign(ctx, worker, task, lane):
    if lane not in LANES:
        raise ToolError(f"lane must be one of {', '.join(LANES)}")
    return ctx.services.team.assign(ctx.agent, worker, task, lane)


def t_team_status(ctx):
    return ctx.services.team.status()


def t_report(ctx, summary, status="done"):
    return ctx.services.team.report(ctx.agent, summary, status)


def _sandboxed(ctx, argv: list[str], timeout_s: int, label: str):
    """argv inside forge.sandbox on the target when the mission asks for it (always on real machines)."""
    m = ctx.mission
    ex = ctx.executor
    if not getattr(m, "sandbox_tools", False):
        return ex.run(argv, cwd=m.ws, timeout_s=timeout_s, mission_id=m.id, label=label)
    venv = posixpath.dirname(posixpath.dirname(m.python))
    wrap = [m.python, "-m", "forge.tools.sbxrun", "--ws", m.ws, "--venv", venv, "--timeout", str(timeout_s)]
    for ro in [m.hf_home, *m.readable_roots]:
        if ro:
            wrap += ["--ro", ro]
    if m.hf_home:
        wrap += ["--env", f"HF_HOME={m.hf_home}", "--env", "HF_HUB_OFFLINE=1"]
    r = ex.run(wrap + ["--"] + argv, cwd=m.forge_root, env={"PYTHONPATH": m.forge_root}, timeout_s=timeout_s + 60,
               mission_id=m.id, label=label)
    if r.rc == 199 and "FORGE_INFRA" in (r.stderr or ""):
        r.infra_error = r.stderr.strip().splitlines()[-1][:300]
    return r


def t_sh(ctx, cmd, timeout_s=300):
    r = _sandboxed(ctx, ["bash", "-c", cmd], min(900, int(timeout_s)), "sh")
    if r.infra_error:
        raise EnvBlocked(f"sh could not run: {r.infra_error}")
    return {"rc": r.rc, "stdout": r.stdout[-4000:], "stderr": r.stderr[-1500:]}


def t_py(ctx, code, timeout_s=300):
    m = ctx.mission
    path = posixpath.join(m.ws, "scratch", f"py_{int(time.time() * 1000)}.py")
    ctx.executor.write_text(path, code)
    r = _sandboxed(ctx, [m.python, path], min(900, int(timeout_s)), "py")
    if r.infra_error:
        raise EnvBlocked(f"py could not run: {r.infra_error}")
    return {"rc": r.rc, "stdout": r.stdout[-4000:], "stderr": r.stderr[-1500:]}


def build_registry() -> Registry:
    reg = Registry()
    add = reg.add
    add(Tool("profile", "Mission target snapshot: recipe serve args, workload, metric, GPU, vLLM/torch versions, "
             "paths, and the reference measurement if one exists.", _obj({}), t_profile, (W, L, V), order=1,
             returns="a dict."))
    add(Tool("bench", "Quick bench on the visible prompts. With no change it measures the reference. With "
             "config_delta (serve flags to set; null removes a flag), env, or bundle it measures that "
             "candidate and its delta_pct against the reference on the same GPU (the reference is cached per "
             "GPU + recipe + prompt set).",
             _obj({"config_delta": O, "env": O, "bundle": S, "n_prompts": I, "fresh_reference": B, "label": S,
                   "claim": S, "predict_pct": {"type": "number"}}),
             t_bench, (W, V), order=2, light=False, returns="reference, candidate, delta_pct, bench_id."))
    add(Tool("read", "Read a file: a venv file by package path (e.g. vllm/...; shows your edited copy if any), "
             "a workspace file, or a model-dir file; directories return a listing.", _obj({"path": S, "start": I, "lines": I}, ["path"]),
             t_read, (W, V), order=10))
    add(Tool("search", "Regex search (grep -E) under a venv package or workspace path.",
             _obj({"pattern": S, "path": S, "max_hits": I, "context": {"type": "integer", "default": 0},
                   "include": S, "ignore_case": B}, ["pattern"]), t_search, (W, V), order=11))
    add(Tool("edit", "Edit a file. Venv files (package path) are edited as a copy in your workspace, never in "
             "place. old+new replaces unique text (with whitespace fallback); replace_all replaces all exact matches; content writes the whole file.",
             _obj({"path": S, "old": S, "new": S, "content": S, "replace_all": B}, ["path"]), t_edit, (W,), order=12))
    add(Tool("apply_patch", "Apply a Begin Patch/End Patch envelope with Add File, Delete File, Update File, optional Move to, and @@ hunks; validate all changes before writing.",
             _obj({"patch": S}, ["patch"]), t_apply_patch, (W,), order=12))
    add(Tool("ls", "List a directory with sizes and edited-copy markers, capped at 300 entries.",
             _obj({"path": {"type": "string", "default": "."}, "depth": {"type": "integer", "default": 1}}),
             t_ls, (W,), order=10))
    add(Tool("glob", "Find files by glob pattern under a path, capped at 200 files.",
             _obj({"pattern": S, "path": {"type": "string", "default": "vllm"}}, ["pattern"]),
             t_glob, (W,), order=11))
    add(Tool("diff", "Unified diff of your edited venv files against the originals.", _obj({"path": S}), t_diff,
             (W, V), order=13))
    add(Tool("revert", "Drop your edited copy of a venv file.", _obj({"path": S}, ["path"]), t_revert, (W,),
             order=14))
    add(Tool("make_bundle", "Freeze a candidate: config_delta, env, your current edits, and kernel files into "
             "a content-addressed bundle.", _obj({"config_delta": O, "env": O, "include_edits": B,
                                                  "kernels": {"type": "array", "items": S}, "note": S}),
             t_make_bundle, (W,), order=15, returns="bundle id and hash."))
    add(Tool("kernel_test", "Run a workspace script on a granted GPU (correctness or microbenchmark).",
             _obj({"path": S, "args": S, "vram_gb": {"type": "number"}, "timeout_s": I}, ["path"]), t_kernel_test,
             (W,), order=16, light=False))
    add(Tool("request_gate", "Ask for a gate run of a bundle. Needs the bench_id of a bench(bundle=...) of that "
             "same bundle. The Verifier reviews it against its bar first.",
             _obj({"bundle": S, "claim": S, "bench_id": S}, ["bundle", "claim", "bench_id"]), t_request_gate,
             (W,), order=20))
    add(Tool("gate_requests", "Gate requests waiting for review, with their bench evidence and the bar.",
             _obj({}), t_gate_requests, (V, L), order=20))
    add(Tool("review_gate", "Approve (sends it to the gate) or reject a gate request, with a reason.",
             _obj({"request_id": S, "approve": B, "reason": S}, ["request_id", "approve", "reason"]),
             t_review_gate, (V,), order=21))
    add(Tool("gate_status", "Gate ledger rows for this mission (verdicts are final only there).", _obj({}),
             t_gate_status, (W, L, V), order=22))
    add(Tool("assign", "Give a worker a task in a lane (replaces its current task).",
             _obj({"worker": S, "task": S, "lane": {"type": "string", "enum": LANES}}, ["worker", "task", "lane"]),
             t_assign, (L,), order=3))
    add(Tool("team_status", "Each worker's task, lane, last results and step count; gate requests.", _obj({}),
             t_team_status, (L,), order=4))
    add(Tool("decide", "Record a decision on the pending triggers: continue | replan | stop | ask_human, with "
             "evidence (bench ids, gate ids, fact ids).",
             _obj({"decision": {"type": "string", "enum": ["continue", "replan", "stop", "ask_human"]},
                   "trigger_ids": {"type": "array", "items": S}, "evidence": S, "plan": S},
                  ["decision", "trigger_ids", "evidence"]), t_decide, (L,), order=5))
    add(Tool("report", "Finish your current task and report to the lead.",
             _obj({"summary": S, "status": {"type": "string", "enum": ["done", "blocked", "failed"]}},
                  ["summary"]), t_report, (W,), order=30))
    add(Tool("note", "Post a short note to the team board.", _obj({"text": S}, ["text"]), t_note, (W, L, V),
             order=31))
    add(Tool("fact_add", "Record a fact in the shared brain (kept across missions).",
             _obj({"claim": S, "lane": {"type": "string", "enum": LANES},
                   "status": {"type": "string", "enum": ["measured", "refuted", "blocked", "open"]},
                   "evidence": S}, ["claim", "lane", "status", "evidence"]), t_fact_add, (W, L, V), order=32))
    add(Tool("facts", "Shared-brain facts for this recipe (optionally one lane or a text query).",
             _obj({"lane": S, "query": S, "limit": I}), t_facts, (W, L, V), order=33))
    add(Tool("ask_human", "Queue a question for a human. Never blocks; keep working.", _obj({"question": S},
             ["question"]), t_ask_human, (W, L, V), order=34))
    add(Tool("sh", "Shell command in your workspace on the target (sandboxed, no GPU).",
             _obj({"cmd": S, "timeout_s": I}, ["cmd"]), t_sh, (W,), order=90))
    add(Tool("py", "Run Python code with the recipe's venv (sandboxed, no GPU).",
             _obj({"code": S, "timeout_s": I}, ["code"]), t_py, (W,), order=91))
    return reg
