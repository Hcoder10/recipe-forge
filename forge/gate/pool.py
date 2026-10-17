"""Hidden prompt pool for the gate.

Built deterministically from a curated public source (the task templates and topics below) and a
secret seed that is created once per machine inside FORGE_ROOT/gate_hidden (mode 700) and never
leaves it. The pool, its slices and the seed are never granted to any sandboxed job.

Strata: task family x prompt length. Each gate rep uses its own fixed slice (one prompt per
stratum, round robin), so every gate sees the same per-rep prompts, no prompt repeats within a server
lifetime, and confirmation gates draw fresh slices from a separate region.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

POOL_VERSION = 1

TOPICS = [
    "tide pools", "bridge design", "sourdough baking", "volcanic soil", "orbital mechanics", "beekeeping",
    "railway signalling", "medieval trade routes", "glacier retreat", "coral reefs", "urban gardening",
    "compiler optimisations", "photosynthesis", "jazz harmony", "desert hydrology", "printing presses",
    "wind turbines", "bird migration", "cave mapping", "chess openings", "lighthouse keeping",
    "textile dyes", "earthquake engineering", "tea cultivation", "radio astronomy", "river deltas",
    "paper making", "solar sails", "mushroom foraging", "clock escapements", "sea ice", "pottery glazes",
    "postal history", "bicycle mechanics", "tidal energy", "library cataloguing", "cheese aging",
    "satellite imaging", "forest fires", "violin making", "salt marshes", "elevator safety",
]

TASKS = {
    "explain": ["Explain {t} to a curious twelve-year-old, with one concrete example.",
                "Give a clear, structured overview of {t}: key ideas, common misconceptions, and open questions.",
                "Why does {t} matter to people who never think about it? Answer in a few paragraphs."],
    "code": ["Write a Python function that simulates a simple model of {t}, with docstrings and a short test.",
             "Write a small command-line tool in Python that keeps a log of observations about {t} in a JSON "
             "file, with add, list and summary commands.",
             "Implement a class hierarchy in Python for records used by a team studying {t}, then show usage."],
    "math": ["A team studying {t} records {a} samples per day for {b} days and loses {c}% of them. Work out how "
             "many usable samples they have, step by step, then generalise the formula.",
             "Set up and solve a word problem about {t} that needs a system of two linear equations. Show work.",
             "Estimate, with stated assumptions, an order-of-magnitude quantity related to {t}. Show the chain "
             "of reasoning."],
    "write": ["Write a short story (about 400 words) in which {t} quietly changes someone's life.",
              "Write a persuasive letter to a city council about funding work on {t}.",
              "Write a dialogue between two experts who disagree about the future of {t}."],
    "extract": ["Read the notes below and produce a JSON object with fields: topic, key_facts (list), "
                "numbers (list of {{value, meaning}}), open_questions (list).\n\nNOTES:\n{doc}",
                "Summarise the following field report in five bullet points, then list anything that looks "
                "inconsistent.\n\nREPORT:\n{doc}",
                "Answer from the document only: what are the three most important measurements, and why?\n\n"
                "DOCUMENT:\n{doc}"],
}
LENGTHS = {"short": 0, "medium": 6, "long": 24}     # paragraphs of context appended / embedded

_WORDS = ("sample site north south reading level gauge morning evening team logged noted rising falling "
          "stable sharp gradual sensor manual estimate measured recorded drift calibration baseline season "
          "storm calm dry wet warm cold variance trend outlier batch cycle phase shift peak trough").split()


def _paragraph(rng: random.Random, topic: str) -> str:
    sents = []
    for _ in range(rng.randint(4, 7)):
        words = [rng.choice(_WORDS) for _ in range(rng.randint(8, 16))]
        n = rng.randint(1, 999)
        sents.append(f"During the {topic} survey the " + " ".join(words) + f" at {n}.{rng.randint(0, 99):02d} units.")
    return " ".join(sents)


def _prompt(rng: random.Random, family: str, length: str, topic: str) -> str:
    tpl = rng.choice(TASKS[family])
    doc = "\n\n".join(_paragraph(rng, topic) for _ in range(max(2, LENGTHS[length])))
    text = tpl.format(t=topic, a=rng.randint(20, 400), b=rng.randint(3, 60), c=rng.randint(2, 30), doc=doc)
    if family != "extract" and LENGTHS[length]:
        ctx = "\n\n".join(_paragraph(rng, topic) for _ in range(LENGTHS[length]))
        text = f"Background material:\n\n{ctx}\n\nTask: {text}"
    return text


def hidden_dir(root: str) -> str:
    d = os.path.join(root, "gate_hidden")
    os.makedirs(d, mode=0o700, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def secret_seed(root: str) -> bytes:
    p = os.path.join(hidden_dir(root), "seed")
    if not os.path.exists(p):
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    with open(p, "rb") as f:
        return f.read()


def build(root: str, per_stratum: int = 12, exclude_texts: list[str] | None = None) -> dict:
    """Build (or load) the pool: {version, sha256, items: [{id, stratum, messages}]}."""
    path = os.path.join(hidden_dir(root), f"pool_v{POOL_VERSION}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    seed = secret_seed(root)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed + b"pool").digest(), "big"))
    excl = {" ".join(t.split()).lower() for t in (exclude_texts or [])}
    items = []
    for family in sorted(TASKS):
        for length in LENGTHS:
            made = 0
            while made < per_stratum:
                topic = rng.choice(TOPICS)
                text = _prompt(rng, family, length, topic)
                if " ".join(text.split()).lower() in excl:
                    continue
                items.append({"id": f"{family}-{length}-{made:02d}", "stratum": f"{family}/{length}",
                              "messages": [{"role": "user", "content": text}]})
                made += 1
    body = json.dumps(items, sort_keys=True).encode()
    pool = {"version": POOL_VERSION, "sha256": hashlib.sha256(body).hexdigest(), "items": items}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(pool, f)
    return pool


class Slices:
    """Fixed, disjoint regions of the pool. Per stratum the items are split into:
    index 0..1 -> warmup, 2..3 -> probes, then rep slices, then confirm slices."""

    def __init__(self, pool: dict, max_reps: int = 8):
        self.by_stratum: dict[str, list[dict]] = {}
        for it in pool["items"]:
            self.by_stratum.setdefault(it["stratum"], []).append(it)
        self.strata = sorted(self.by_stratum)
        self.max_reps = max_reps

    def _take(self, offset: int, count: int, short_only: bool = False) -> list[dict]:
        out = []
        for s in self.strata:
            if short_only and not s.endswith("/short"):
                continue
            items = self.by_stratum[s]
            out += items[offset: offset + count]
        return out

    def warmup(self) -> list[dict]:
        return self._take(0, 1, short_only=True)

    def probes(self) -> list[dict]:
        return [it for it in self._take(1, 1) if not it["stratum"].endswith("/short")]

    def _ordered(self, offset: int, n: int) -> list[dict]:
        """n prompts round robin over strata, starting at per-stratum index `offset`."""
        out, k = [], 0
        while len(out) < n:
            s = self.strata[k % len(self.strata)]
            idx = offset + k // len(self.strata)
            items = self.by_stratum[s]
            if idx >= len(items):
                raise ValueError("pool too small for this slice")
            out.append(items[idx])
            k += 1
        return out

    def rep(self, r: int, n: int) -> list[dict]:
        """Prompts for rep r of any candidate/reference gate (the same for every gate)."""
        per = -(-n // len(self.strata))
        return self._ordered(2 + r * per, n)

    def confirm(self, c: int, rep: int, n: int) -> list[dict]:
        """Fresh prompts for confirmation c (2 reps each), after the rep region."""
        per = -(-n // len(self.strata))
        base = 2 + self.max_reps * per
        return self._ordered(base + (c * 2 + rep) * per, n)
