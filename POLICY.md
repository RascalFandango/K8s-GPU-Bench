# POLICY.md — NRP/Nautilus compliance assessment

Compliance reference for this project. **Current NRP documentation always wins**
over this file, the BUILD-SPEC, and the README. Re-verify before trusting — NRP
changes policy without notice.

- **Assessed:** 2026-08-25
- **Sources (verify these, not this summary):**
  - Acceptable Use Policy (AUP): <https://nrp.ai/NRP-AUP.pdf>
  - Cluster Policies: <https://nrp.ai/documentation/userdocs/start/policies/>
  - Priority Classes / Opportunistic: <https://nrp.ai/documentation/userdocs/running/priority-classes/>
  - Special Use (taints): <https://nrp.ai/documentation/userdocs/running/special/>
  - GPU Pods (resource names, quota): <https://nrp.ai/documentation/userdocs/running/gpu-pods/>
  - Batch Jobs: <https://nrp.ai/documentation/userdocs/running/jobs/>
  - Monitoring: <https://nrp.ai/documentation/userdocs/running/monitoring/>
  - Long-Idle Pods: <https://nrp.ai/documentation/userdocs/running/long-idle/>

## Verdict

The project is **viable and not fundamentally against the Terms.** Enumerating
GPU nodes — the activity we were unsure about — is explicitly endorsed: the docs
tell users to run `kubectl get nodes -L nvidia.com/gpu.product` to choose
hardware. The AUP's only nearby clause bars attempts to *"breach or circumvent
any NRP administrative controls or security controls"*, and reading node metadata
you have granted read access to is neither.

Five design corrections were required; all are implemented or scheduled below.

## Compliance matrix

| Component | Verdict |
|---|---|
| Node scraper (reads node labels) | ✅ Clean — recommended practice |
| Benchmark exits on its own (no `sleep infinity`) | ✅ Clean |
| Monitoring = Deployments, GPU work = Jobs | ✅ Clean |
| Self-hosted Prometheus/Grafana | ✅ Clean — not duplicating enforcement |
| Monitoring resource requests (<1 CPU / <2 GB each) | ✅ Exempt from usage-violation rule |
| Auto-generated tolerations | ⚠️ Fixed — see F1 |
| Fleet-wide Job submission | ⚠️ Scheduled — see F2 (Phase 5) |
| GPU utilization on bench Jobs | ⚠️ Design rule — see F3 (Phase 3/5) |
| Special-resource names / reserved HW | ⚠️ Fixed — see F4 |
| `--opportunistic` for gated GPUs | ✅ Clean — see F5 (spec was wrong) |

## Findings

### F1 — Tolerations must be an authorization allowlist *(fixed in Phase 1)*
Special Use defines taint tiers; users may tolerate only what they're authorized
for. The scraper now classifies taints and auto-proposes tolerations ONLY for
`nautilus.io/hardware`. All other non-churn taints are surfaced as `restricted`
and never auto-tolerated. Taint scheme also migrated:
`nautilus.io/arm64=true` → `nautilus.io/hardware=arm64:NoSchedule`.

Tiers:
| Taint key | Tolerable? |
|---|---|
| `nautilus.io/hardware` (arm64, large-gpu) | ✅ any user |
| `nautilus.io/science-dmz` | ⚠️ only if no public-Internet need — treated as restricted |
| `nautilus.io/reservation` | ⛔ only if in the approved group |
| `nautilus.io/system` | ⛔ never |
| `nautilus.io/issue` | ⛔ never |
| `node.kubernetes.io/*` | churn — ignored |

### F2 — No fair queue; do not flood *(scheduled: Phase 5)*
Batch Jobs page: *"If you submit 1000 jobs, you block all other users."*
`submit.py` must throttle (submit a few flavors at a time); default posture is a
handful of flavors, never the whole fleet.

### F3 — Keep requested GPUs ≥40% utilized *(design rule: Phase 3/5)*
A user with >4 pods each under 40% GPU utilization is bannable. GPU-requesting
Jobs must actually exercise the GPU; keep `PRESCRAPE_DELAY`/`POSTSCRAPE_DELAY`
short relative to compute; never spray many `inventory`-only GPU Jobs at once.

### F4 — Correct resource names + reserved hardware *(fixed)*
Real names (2026-08): `a100`, `h100`, `h200`, `gh200`, `a40`, `rtxa6000`,
`rtx8000` (no dash — BUILD-SPEC fixture had `rtx-8000`), `rtx6000bw`,
`mig-small`. Derivation (§2.6) handles whatever a node advertises, so production
is correct regardless; fixture and docs corrected for accuracy. RTX PRO 6000
Blackwell nodes are reserved / not generally available — flag, do not submit.

### F5 — `opportunistic` is sanctioned, not a circumvention *(spec corrected)*
BUILD-SPEC said to use the A100 form "rather than route around the control" with
opportunistic. NRP docs explicitly document opportunistic as a legitimate path
for A100/H100/H200/GH200: it *"bypasses the GPU quota but can be preempted at any
time."* The A100 form grants *reserved* (non-preemptible) quota;
H100/H200/GH200 have **no form** — opportunistic is the only user path. Keep
`--opportunistic`. Allowed priority classes (admission-enforced): unset,
`armada-default`, `owner-no-preempt`, `opportunistic`, `opportunistic2`. Any
other is rejected — never emit one.

## Other AUP notes
- **No protected data** (HIPAA/PII/CUI/FERPA). We store only hardware metrics — fine.
- **Not for financial gain.** Benchmarking to inform *your* university's
  purchasing is research use and fine; running this as a paid service is not.
- **Acknowledge NRP** in any resulting publication (AUP §7).
- **Storage:** volumes untouched for 6 months may be purged without notice.
- **Ephemeral storage:** a container writing >50Gi scratch can be evicted; set
  `ephemeral-storage` requests on bench Jobs if they spill to scratch.
