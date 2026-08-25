# CLAUDE.md — nrp-gpu-bench

Project conventions and hard constraints. Read before making changes.

## What this is

A GPU discovery, benchmarking, and comparison toolkit for the National Research
Platform ("Nautilus") Kubernetes cluster. Enumerates every GPU flavor on the
cluster, benchmarks each, and stores results in a self-hosted
Prometheus/Grafana stack for side-by-side comparison.

## Policy precedence — read this first

**Current NRP documentation always wins.** Where this file, the BUILD-SPEC, or
the README conflict with the live policy at https://nrp.ai/documentation/, the
NRP docs are authoritative and the conflict is resolved in their favor. Several
BUILD-SPEC details were already found stale against the docs (taint scheme,
special-resource names, the opportunistic guidance) and corrected here. See
[POLICY.md](POLICY.md) for the full compliance assessment, source links, and the
date each rule was verified. Re-verify before trusting; NRP changes.

## Hard constraints — do not violate

- **No non-terminating containers.** A Job running `sleep` or any command that
  never exits is a bannable offense on this cluster. Every workload must exit
  on its own. (Bounded `time.sleep()` mid-program is fine; only a *script that
  ends in sleep* / `sleep infinity` is banned.)
- **Tolerations are an allowlist, never blanket.** Users may tolerate only
  taints they are authorized for. Auto-tolerate ONLY the `nautilus.io/hardware`
  tier (arm64, large-gpu). NEVER auto-tolerate `nautilus.io/reservation` (needs
  group membership), `nautilus.io/system` or `nautilus.io/issue` (never), or
  `nautilus.io/science-dmz` (conditional). Tolerating a taint you lack
  authorization for is itself a policy violation. The scraper classifies taints;
  `submit.py` must only emit `hardware`-tier tolerations.
- **Do not flood the queue.** There is no fair queue on Nautilus — a bulk
  submission blocks every other user. `submit.py` must throttle (submit a few
  flavors at a time), and the default posture is a handful of flavors, never the
  whole fleet.
- **Keep requested GPUs busy (≥40% utilization).** A user with more than 4 pods
  each below 40% GPU utilization is bannable. GPU-requesting Jobs must actually
  exercise the GPU; keep pre/post-scrape idle windows short, and never spray many
  `inventory`-only (near-zero-util) GPU Jobs at once.
- **Only these priority classes are allowed:** unset (default), `armada-default`,
  `owner-no-preempt`, `opportunistic`, `opportunistic2`. Any other
  `priorityClassName` is rejected by admission — `submit.py` must never emit one.
- **`opportunistic` is the sanctioned no-reservation path to A100/H100/H200/**
  **GH200.** It bypasses the per-namespace quota in exchange for preemption, and
  NRP documents it explicitly — it is NOT a circumvention. The A100 form grants
  *reserved* (non-preemptible) quota; H100/H200/GH200 have no form at all, so
  opportunistic is the only user path. Keep `--opportunistic`.
- **Namespace-scoped access only.** No ClusterRoles, no CRDs, no Operators, no
  cluster-wide ServiceAccount permissions. They will fail to apply.
- **Namespace-scoped access only.** No ClusterRoles, no CRDs, no Operators, no
  cluster-wide ServiceAccount permissions. They will fail to apply.
- **No Helm, no Prometheus Operator.** Plain manifests plus Kustomize. This
  follows from the constraint above.
- **Deployments are purged after 2 weeks** unless the namespace has an
  exception. PVCs survive, so re-applying restores state — design for that,
  don't fight it.
- **GPU-requesting workloads must be Jobs, never Deployments.** Deployments
  cannot request GPUs here.
- **Never hardcode a list of GPU models.** Special-request flavors are derived
  from the node's allocatable map: any `nvidia.com/*` resource that isn't
  exactly `nvidia.com/gpu`. Hardcoded lists rot as NRP adds hardware. (For
  reference only — real names as of 2026-08: `a100`, `h100`, `h200`, `gh200`,
  `a40`, `rtxa6000`, `rtx8000` (no dash), `rtx6000bw`, `mig-small`. RTX PRO 6000
  Blackwell nodes are reserved and not generally available — flag, don't submit.)

## Load-bearing details that look optional but aren't

- `honor_labels: true` on the Pushgateway scrape config. Without it, Prometheus
  overwrites the pushed `product`/`node` labels and every GPU flavor collapses
  into one series. Silent failure — the dashboard just shows identical numbers.
- Pushgateway grouping key is `(product, node)`. Adding a run ID or timestamp
  gives unbounded cardinality growth.
- Pod scrape interval is 5s, not the 15s default. Benchmark pods are short-lived
  and the default can miss a run entirely.
- `torch.backends.cuda.matmul.allow_tf32` must be reset in a `finally` block, or
  the fp32 and tf32 measurements come out identical.
- GPU timing uses CUDA events with a warmup, never wall clock. Kernel launches
  are asynchronous.
- Pushgateway needs `--persistence.file`, or a pod restart loses every result.

## Conventions

- Python: stdlib-first, minimal dependencies. Scripts take input on stdin by
  default so they work from a laptop with the operator's own credentials —
  this deliberately sidesteps the ServiceAccount node-read RBAC problem.
- Every benchmark is individually guarded. One unsupported dtype on an old card
  records a skip and the run continues; it never aborts the whole run.
- Metrics are prefixed `nrpbench_`.
- Manifests are numbered by apply order (`10-`, `20-`, `30-`).
- Secrets are placeholders in git. Never commit a real credential.

## Testing without a cluster

`inventory/testdata/nodes.json` is a synthetic fixture covering the edge cases:
multi-GPU node, cordoned node, special-resource node (`rtx8000`), a
`hardware=arm64`-tainted node (tolerable), a `reservation`-tainted node
(restricted — must NOT be auto-tolerated), and a CPU-only node that must be
skipped. Expected: **4 flavors across 5 GPU nodes.** The inventory scraper and
Job generator are both fully testable against it.

For `bench.py`, verify that calling a guarded benchmark without torch installed
returns `None` rather than raising, and that `push_results()` against a dead
address fails with a network error (`URLError`) rather than a `ValueError` —
the former proves every Prometheus label set constructed correctly.

## Context

The operator is an experienced HPC/RHEL sysadmin using this project to build
Kubernetes and observability skill. When a Kubernetes concept maps onto
something from bare-metal HPC — Jobs to batch submission, taints/tolerations to
SLURM partition constraints, RBAC to account/QoS restrictions — say so.
Explanations should assume deep systems knowledge and no Kubernetes fluency.

## Verify before trusting

NRP policy details change. Check https://nrp.ai/documentation/ rather than
relying on what's written here or in the README. [POLICY.md](POLICY.md) records
the full assessment and the exact pages/dates each rule was verified against;
refresh it when policy moves. On any conflict, the live docs win (see top).
