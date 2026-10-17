"""Lease keeper: rents, extends and moves out landlord leases, and never forgets one.

State is persisted after every change, so a crash or a landlord outage loses nothing: leases that
could not be moved out stay in `pending_out` and are retried on every tick and on the next start.
"""
from __future__ import annotations

import json
import os
import threading
import time

from forge.sched.landlord import LandlordClient, LandlordError, LandlordUnreachable

EXTEND_WHEN_LEFT_S = 12 * 60     # extend once less than this is left
EXTEND_BY_MIN = 30


def _expiry(lease: dict, now: float) -> float | None:
    """Expiry on our own clock. The landlord's timestamps are in the coordinator's local zone."""
    left = lease.get("minutes_left")
    try:
        return now + float(left) * 60 if left is not None else None
    except (TypeError, ValueError):
        return None


class LeaseKeeper:
    def __init__(self, state_path: str, client: LandlordClient | None = None, building: str | None = None,
                 clock=time.time):
        self.state_path = state_path
        self.client = client or LandlordClient()
        self.building = building
        self.clock = clock
        self.lock = threading.RLock()
        self.leases: dict[str, dict] = {}       # lease_id -> {gpu, grant, purpose, expires, env, rented}
        self.pending_out: dict[str, str] = {}   # lease_id -> outcome
        self.last_error: str | None = None
        self.last_ok: float | None = None
        self._load()

    # -- persistence -------------------------------------------------------------
    def _load(self):
        try:
            with open(self.state_path) as f:
                st = json.load(f)
            self.leases = st.get("leases", {})
            self.pending_out = st.get("pending_out", {})
        except (OSError, ValueError):
            pass

    def save(self):
        with self.lock:
            tmp = self.state_path + ".tmp"
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump({"leases": self.leases, "pending_out": self.pending_out}, f, indent=1)
            os.replace(tmp, self.state_path)

    def _ok(self):
        self.last_ok = self.clock()
        self.last_error = None

    # -- operations --------------------------------------------------------------
    def rent(self, gpu: int, pct: int, vram_gb: float, minutes: int, purpose: str, grant: str) -> dict:
        """Rent one GPU. Raises LandlordUnreachable (retry later) or LandlordError (refused)."""
        try:
            r = self.client.rent(pct=pct, vram_gb=vram_gb, purpose=purpose, minutes=minutes,
                                 building=self.building, gpu=gpu)
        except LandlordUnreachable as e:
            self.last_error = f"unreachable: {e}"
            raise
        self._ok()
        lease = r.get("lease", r) if isinstance(r, dict) else {}
        lid = lease.get("id")
        if not lid:
            raise LandlordError(f"rent returned no lease id: {str(r)[:300]}")
        got = lease.get("gpu_idxs") or [gpu]
        env = lease.get("env") or (r.get("launch_with") or {}).get("env") or {}
        rec = {"gpu": got[0], "grant": grant, "purpose": purpose, "pct": pct, "vram_gb": vram_gb,
               "expires": _expiry(lease, self.clock()) or self.clock() + minutes * 60,
               "env": env, "rented": self.clock(), "extensions": 0}
        with self.lock:
            self.leases[lid] = rec
            self.save()
        if got[0] != gpu:     # landlord placed us elsewhere: give it back
            self.release(lid, f"wrong GPU {got[0]} (wanted {gpu}); returned unused")
            raise LandlordError(f"landlord placed lease on GPU {got[0]}, wanted {gpu}")
        return {"lease_id": lid, **rec}

    def release(self, lease_id: str, outcome: str) -> bool:
        """Move out now if possible; otherwise queue the move-out. Returns True if done."""
        with self.lock:
            self.leases.pop(lease_id, None)
            self.pending_out[lease_id] = outcome
            self.save()
        return self._flush_one(lease_id)

    def _flush_one(self, lease_id: str) -> bool:
        outcome = self.pending_out.get(lease_id)
        if outcome is None:
            return True
        try:
            self.client.move_out(lease_id, outcome)
        except LandlordUnreachable as e:
            self.last_error = f"unreachable: {e}"
            return False
        except LandlordError as e:
            # already gone/expired: nothing left to free
            self.last_error = f"move_out {lease_id}: {e}"
        else:
            self._ok()
        with self.lock:
            self.pending_out.pop(lease_id, None)
            self.save()
        return True

    def tick(self, active_grants: set[str], notes: dict[str, str] | None = None) -> None:
        """Retry move-outs, extend leases whose grant is still active, drop orphans."""
        for lid in list(self.pending_out):
            if not self._flush_one(lid):
                break
        now = self.clock()
        for lid, rec in list(self.leases.items()):
            if rec["grant"] not in active_grants:
                self.release(lid, "grant ended; lease returned")
                continue
            if rec["expires"] - now < EXTEND_WHEN_LEFT_S:
                note = (notes or {}).get(rec["grant"]) or f"still running: {rec['purpose'][:120]}"
                try:
                    r = self.client.extend(lid, EXTEND_BY_MIN, note)
                except LandlordUnreachable as e:
                    self.last_error = f"unreachable: {e}"
                    continue
                except LandlordError as e:
                    self.last_error = f"extend {lid}: {e}"
                    if rec["expires"] < now:      # truly gone
                        with self.lock:
                            self.leases.pop(lid, None)
                            self.save()
                    continue
                self._ok()
                lease = r.get("lease", r) if isinstance(r, dict) else {}
                with self.lock:
                    rec["expires"] = _expiry(lease, self.clock()) or rec["expires"] + EXTEND_BY_MIN * 60
                    rec["extensions"] += 1
                    self.save()

    def close(self, outcome: str = "scheduler shutting down; all work stopped", deadline_s: float = 45.0) -> bool:
        """Move out every lease. Retries until deadline; whatever is left stays persisted."""
        with self.lock:
            for lid in list(self.leases):
                self.pending_out[lid] = outcome
            self.leases.clear()
            self.save()
        end = time.time() + deadline_s
        while self.pending_out and time.time() < end:
            for lid in list(self.pending_out):
                self._flush_one(lid)
            if self.pending_out:
                time.sleep(2)
        return not self.pending_out

    def snapshot(self) -> dict:
        with self.lock:
            return {"leases": json.loads(json.dumps(self.leases)), "pending_out": dict(self.pending_out),
                    "last_error": self.last_error, "last_ok": self.last_ok}
