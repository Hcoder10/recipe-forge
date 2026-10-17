# Results (Oct 17 2026, Recursive Agents Hackathon)

Every number below comes from the harness's own ledgers: `state/missions/<id>/benches.jsonl` and the
gate ledger `gate/ledger.jsonl` on the machine that ran it. A quick bench is not a verified win. Only a
gate verdict is.

## Candidates found by the Nemotron agents

Recipe: **NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB + LiLiCorr drafter**, NVIDIA's published vLLM command,
single stream (`bolt-74b-lilicorr:c1`), on an RTX PRO 6000 Blackwell Max-Q (rig).
Mission `1017-143651-bolt-74b-lilicorr-c1-rig`: a team of Nemotron 3.5 Super-VL agents on Crusoe, a
75 min time box, $0.38 of LLM spend.

| Bench | Change (found by the agents) | Candidate tok/s | Fresh reference tok/s | Delta | Reference noise | Verdict hint | Bundle | Gate |
|---|---|---|---|---|---|---|---|---|
| B13 | `--async-scheduling` | 325.5 | 314.9 | **+3.68%** | 0.63% | gain (confirmed) | `720eb19ae50c6d74` | `g10171554-80f05a`: **LOSS**, mean -0.97%, 99.9% CI on log-gain [-6.7%, +4.7%] (df_null 8). The quick-bench gain did not survive paired AB/BA timing. |
| B14 | `--attention-backend TRITON_ATTN` | 313.9 | 305.6 | **+3.38%** | 1.28% | gain (confirmed) | `e40ad4fce88f7f2b` | `g10171554-d1f899`: PENDING |

"Confirmed" means the bench re-measured the reference in the same job after the candidate (A/B/A), and
the delta is computed against that fresh measurement.

**Disclosure.**
- The agents found both changes, measured them and froze them into bundles themselves.
- The mission's time box ran out one minute before the Verifier submitted them. The Lead's last decision
  reads: "Two candidates cleared the +3.0% bar but no gate request was ever submitted". The operator then
  submitted the two unmodified agent bundles to the gate (`requester: operator/late-submit`).
- This mission received one human steering note, about the drafter, which the agents did not end up
  pursuing. The note text is in the mission's `operator_notes.jsonl`.

## Refused by the harness (Done Means Done)

| Mission | Shown | Real | Why refused |
|---|---|---|---|
| Super-VL rig `1017-142354` B4, `--mamba-cache-philox-rounds 3` | +8.2% vs a 9-min-old cached reference | +1.1% vs a fresh reference | `within_noise` (reference noise 6.6%) |
| Earlier missions (m1b, m2) | +5% from a single cached reference | +0.6% on re-measure | quick-bench noise across server restarts, 4-5% |

## Gate validation

- Reference-vs-reference controls on Bolt c1 hold 8 null pairs (`g10171334-bd7ba0`).
- Two no-op plumbing candidates came back `loss`, with the lossless check passing (`g10171422-694b18`,
  `g10171443-13dd8f`), about 18 min each.
