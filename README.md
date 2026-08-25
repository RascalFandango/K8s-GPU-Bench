# nrp-gpu-bench

A GPU discovery, benchmarking, and comparison toolkit for the National Research
Platform ("Nautilus") Kubernetes cluster. It enumerates every GPU flavor on the
cluster, benchmarks each, and stores the results in a self-hosted
Prometheus + Grafana stack so you can compare cards side by side with **measured
numbers** — for skill-building, and to justify hardware purchases with data
instead of vendor spec sheets.

NRP already runs its own Grafana with per-namespace GPU **utilization**
dashboards. This project does **not** rebuild that. It answers a different
question theirs doesn't: *cross-flavor comparison with durable results.*

> **Policy first.** Everything here obeys current NRP policy, and where the
> original build spec conflicted with the live docs, **the docs win**. See
> [POLICY.md](POLICY.md) for the full compliance assessment and
> [CLAUDE.md](CLAUDE.md) for the encoded constraints. Re-verify against
> <https://nrp.ai/documentation/> before trusting anything policy-related.

## The two-path design

Benchmark Jobs are ephemeral — a pod that lives twelve minutes might be scraped
twice or zero times, and its time series vanish when it's deleted. So metrics
travel two paths:

| Path | Carries | Mechanism | Durable? |
|---|---|---|---|
| **Live telemetry** | utilization, power, temp, SM clock, memory | NVML sampled in-process, exposed on `:9400`, scraped every 5s | No — and that's correct, it's only meaningful during the run |
| **Results** | TFLOPS, GB/s, tokens/sec | pushed to the Pushgateway when the run finishes, keyed by `(product, node)` | **Yes** |

This is the canonical Pushgateway pattern (ephemeral batch jobs), and it's the
single most useful Prometheus idea in the project.

## Repository layout

```
nrp-gpu-bench/
├── CLAUDE.md                     encoded hard constraints (auto-read each session)
├── POLICY.md                     NRP compliance assessment + source links
├── README.md                     this file
├── inventory/
│   ├── gpu_inventory.py          node scraper -> flavor catalog (stdlib only)
│   ├── cronjob.yaml              weekly refresh (leads with the RBAC trap)
│   └── testdata/nodes.json       synthetic fixture, 5 edge cases
├── bench/
│   ├── bench.py                  NVML exporter + benchmarks + Pushgateway push
│   ├── submit.py                 flavor catalog -> Job manifests
│   └── Dockerfile                multi-arch (amd64 + arm64) benchmark image
└── k8s/
    ├── kustomization.yaml
    ├── 10-prometheus.yaml        RBAC + config + PVC + Deployment + Service
    ├── 20-pushgateway.yaml       persisted Deployment + PVC + Service
    ├── 30-grafana.yaml           datasource/dashboard provisioning + Deployment
    └── dashboards/gpu-comparison.json   9-panel comparison dashboard
```

## Prerequisites

- `kubectl`, and a Nautilus namespace where you have **namespace admin**.
- Cluster-wide **node read** with your *own* user credentials (you already have
  this; a ServiceAccount does not — see the CronJob's RBAC note).
- `python3` (standard library only — no pip installs for the scraper/submitter).
- A container registry you can push to, and `docker`/`buildx` (or `podman`) to
  build the benchmark image.
- Confirm your storage class before applying (manifests assume `rook-ceph-block`):

```bash
kubectl get storageclass
```

## Quickstart

```bash
# 1. Target your namespace (either set it in k8s/kustomization.yaml's
#    `namespace:` field, or pass -n on every command / set your context).
kubectl config set-context --current --namespace=<your-namespace>

# 2. Set a real Grafana admin password (the committed value is CHANGE-ME).
kubectl create secret generic grafana-admin \
  --from-literal=admin-password='<pick-a-password>' \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Deploy the monitoring stack.
kubectl apply -k k8s/

# 4. See what GPUs exist (uses YOUR node-read creds, from your laptop).
kubectl get nodes -o json | python3 inventory/gpu_inventory.py --format md

# 5. Build and push the benchmark image (multi-arch for the arm64 GH200 nodes).
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <registry>/nrp-gpu-bench:latest --push bench/

# 6. Benchmark ONE flavor with the cheap tiers first (see "The first run").
kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py --image <registry>/nrp-gpu-bench:latest \
      --only 3090 --tiers inventory,micro \
  | kubectl apply -f -

# 7. Watch it, then view the dashboard.
kubectl logs -l app=nrp-gpu-bench -f
kubectl port-forward svc/grafana 3000:3000   # http://localhost:3000
```

## What gets measured (per tier)

Pass any subset via `--tiers` (submit) / `BENCH_TIERS` (env).

- **Tier 0 — `inventory`**: device name, compute capability, VRAM, SM count,
  torch/CUDA/driver versions, enforced power limit, PCIe generation and width.
  Near-zero GPU work — don't run this tier alone in bulk (see the 40% gotcha).
- **Tier 1 — `micro`**:
  - **GEMM** — square matmul TFLOPS, gated by compute capability (fp32 always;
    tf32/bf16 need CC ≥ 8.0; fp16 needs CC ≥ 7.0).
  - **Memory bandwidth** — STREAM triad, GB/s.
  - **PCIe** — pinned host↔device copy bandwidth, GB/s.
- **Tier 2 — `workload`**: a real `TransformerEncoderLayer` training step
  (forward + backward + AdamW), reported as **tokens/sec** — the number
  researchers actually care about.

All result metrics are prefixed `nrpbench_` and carry `(product, node)`.

## Interpreting results

The raw numbers mislead if you read them in the wrong order.

- **Read memory bandwidth before TFLOPS.** Most research codes — and *all* LLM
  inference — are memory-bound, not compute-bound. A card with huge FP32 TFLOPS
  and mediocre bandwidth will disappoint on real workloads.
- **A throttled run is a floor, not a ceiling.** Check the temperature + SM-clock
  panel: a clock that sags as temperature climbs is thermal throttling. That
  card can do better with more cooling — the number you measured is the worst
  case for that chassis, not the silicon's limit.
- **One Job measures one node.** Cooling and PCIe topology vary between nodes of
  the same flavor. Run several Jobs per flavor and compare; `max by (product)`
  in the dashboard keeps the best observed result.
- **TFLOPS/W is the column that matters for purchasing.** The university pays for
  power and cooling too. The comparison matrix computes bf16 TFLOPS ÷ TDP with a
  gradient — that's the efficiency ranking, and it often reorders the raw-TFLOPS
  leaderboard.

## The first run, narrated

**Start with one flavor and the cheap tiers.** Do not submit the full fleet on
the first run — there is no fair queue on Nautilus (see gotchas).

```bash
# Deploy the stack. Watch the three Deployments come up.
kubectl apply -k k8s/
kubectl get pods -w
#   Pending           -> waiting for a node / PVC to bind
#   ContainerCreating -> pulling images, mounting volumes
#   Running           -> serving. Ctrl-C when all three are Running.

# Open Grafana (login: admin / the password from Quickstart step 2).
kubectl port-forward svc/grafana 3000:3000     # http://localhost:3000

# See the catalog. This uses YOUR credentials from the laptop.
kubectl get nodes -o json | python3 inventory/gpu_inventory.py --format md

# Build + push the image (once), then submit ONE flavor, cheap tiers:
kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py \
      --image <registry>/nrp-gpu-bench:latest \
      --only 3090 --tiers inventory,micro \
  | kubectl apply -f -

# Follow the run. Expect the pre-scrape sleep, then benchmarks, then the push.
kubectl logs -l app=nrp-gpu-bench -f
```

Panels stay empty until the first results push and Prometheus scrapes them.
Give it a minute after the log says `results pushed`. Widen `--only` and add
`--tiers workload` once one flavor works end to end.

### When something is wrong (and it will be)

```bash
# The Job's own status and its pod's events (read Events first — bottom up):
kubectl describe job/<job-name>
kubectl describe pod/<pod-name>          # "Insufficient nvidia.com/gpu", taints, image pull

# Everything that happened recently, newest last:
kubectl get events --sort-by=.lastTimestamp

# Logs from a crashed previous attempt:
kubectl logs <pod-name> --previous

# Did Prometheus actually discover the bench pod? Check its scrape targets:
kubectl port-forward svc/prometheus 9090:9090   # http://localhost:9090/targets
```

Common first-run causes: pod `Pending` because the flavor is quota-gated (use
`--opportunistic`) or tainted (arm64 needs a multi-arch image); empty dashboard
because the run finished before Prometheus scraped it (that's what the
post-scrape delay is for) or because the image wasn't pushed for the node's arch.

## Gotchas

- **Deployments are purged after 2 weeks.** The monitoring stack is Deployments;
  they'll be deleted unless your namespace has an exception (ask Nautilus
  Support for a permanent service). **PVCs survive**, so `kubectl apply -k k8s/`
  restores everything — design for re-applying, don't fight it.
- **Nothing may run forever.** A Job with `sleep infinity` (or a script ending in
  sleep) is a bannable offense. Every workload here exits on its own; the bench's
  pre/post-scrape delays are bounded sleeps inside a program that then exits.
- **`honor_labels: true`** on the Pushgateway scrape is load-bearing. Without it,
  Prometheus overwrites the pushed `product`/`node` labels and every flavor
  collapses into one series — the dashboard silently shows identical numbers.
- **allocatable ≠ free.** The inventory shows what GPUs *exist* and whether their
  nodes are schedulable — not what's idle right now (a namespace can't see
  cluster-wide allocation). Check NRP's own Grafana for live utilization.
- **Special-request GPUs are requested by resource name**, derived from the node,
  never hardcoded: `nvidia.com/a100`, `h100`, `h200`, `gh200`, `a40`,
  `rtxa6000`, `rtx8000` (no dash), `rtx6000bw`, `mig-small`. RTX PRO 6000
  Blackwell nodes are reserved and not generally available.
- **Opportunistic vs the A100 form.** A100/H100/H200/GH200 sit behind a
  per-namespace quota that defaults to zero. `submit.py --opportunistic` is the
  **sanctioned** no-reservation path (preemptible; the quota doesn't apply). The
  A100 *form* grants *reserved*, non-preemptible quota; H100/H200/GH200 have no
  form, so opportunistic is the only user path. Only `opportunistic` /
  `opportunistic2` (and the defaults) pass admission — never set another class.
- **Tolerations are an allowlist.** Only `nautilus.io/hardware` (arm64,
  large-gpu) is auto-tolerated. Reservation / system / issue taints are
  **never** auto-tolerated — the scraper marks those flavors `restricted` and
  `submit.py` skips them. GH200 also needs an **arm64 image** (build multi-arch).
- **No fair queue.** Submitting hundreds of Jobs blocks every other user.
  `submit.py --max-jobs` caps output (default 5); raise it deliberately and
  narrow with `--only`.
- **Keep requested GPUs ≥40% utilized.** More than 4 pods each below 40% GPU
  utilization is bannable. Don't spray `inventory`-only (near-zero-util) GPU Jobs.
- **Verify the storage class.** `rook-ceph-block` is an assumption
  (`kubectl get storageclass`).
- **Change the Grafana password.** The committed secret is a `CHANGE-ME`
  placeholder.

## Testing without a cluster

The scraper and Job generator are fully testable against the synthetic fixture
(no cluster, no GPU):

```bash
# 4 flavors across 5 nodes; the reserved flavor is skipped by submit.py.
python3 inventory/gpu_inventory.py --file inventory/testdata/nodes.json --format md
python3 inventory/gpu_inventory.py --file inventory/testdata/nodes.json --format json \
  | python3 bench/submit.py --image test:latest --tiers inventory,micro
```

`bench.py` compiles and imports without torch/pynvml; its guarded benchmarks
return `None` and record a skip rather than raising, and `push_results()`
against a dead address fails with a network error (proving every Prometheus
label set is valid). See [CLAUDE.md](CLAUDE.md) for the exact checks.

## Stretch goals

Archiving results to Ceph S3 for retention-independent history; multiple nodes
per flavor to expose within-flavor variance; an LLM inference tier (vLLM /
llama.cpp tokens/sec); a Markdown report generator for purchase justifications;
an arm64 image benchmarked on GH200.
