# nrp-gpu-bench — Deployment & Usage Guide

A complete, copy-paste, step-by-step guide to deploying and running this toolkit
on the NRP/Nautilus cluster: deploy the monitoring stack, build the image,
discover GPU flavors, and run benchmarks tier by tier. See also the shorter
[README](README.md) and the compliance notes in [POLICY.md](POLICY.md).

> **Portability (is this NRP-only?).** The **benchmark core** (`bench.py` —
> CUDA-event timing, GEMM/bandwidth/PCIe/Transformer, NVML telemetry) and the
> **two-path metric design** (scrape live + push results) are universal and run
> on any Kubernetes cluster with NVIDIA GPUs. Much of the *deployment*
> architecture, however, is shaped by NRP's constraints and is **NRP-specific**:
> the taint tiers (`nautilus.io/hardware|reservation|…`), the priority classes
> (`opportunistic`), the special resource names (`nvidia.com/a100`, `rtx8000`,
> …), the `rook-ceph-block` storage class, and the whole "namespace-scoped, no
> ClusterRole, no Operator/Helm" shape (which exists *because* NRP denies cluster
> RBAC). On a cluster where you have cluster-admin you would keep `bench.py` but
> replace `k8s/` with the NVIDIA GPU Operator + Prometheus Operator
> (`ServiceMonitor`) + Helm. It also assumes **GPU Feature Discovery** node
> labels (`nvidia.com/gpu.product`, `.memory`, `.compute.*`) exist; without GFD
> there is no product/VRAM/compute data to scrape.

> **Mental model first (read this).** "Getting the repo onto the cluster" is
> three different things, not one:
>
> | Piece | Where it lives | How it gets there |
> |---|---|---|
> | `k8s/` manifests (Prometheus/Grafana/Pushgateway) | on the cluster as objects | `kubectl apply -k k8s/` |
> | `bench/bench.py` (the benchmark) | on the cluster **inside a container image** | you build + push an image; Jobs pull it |
> | `inventory/gpu_inventory.py`, `bench/submit.py` | **your laptop** | run locally; they emit YAML you pipe to kubectl |
>
> The Python tools run **on your machine with your own credentials** (that's the
> whole stdin-first design). Only the monitoring stack and the benchmark Jobs
> actually run on the cluster.

---

## Conventions used below

```bash
# Set these once per shell session and reuse them everywhere.
export NS=<your-nautilus-namespace>              # e.g. slu-eus
export IMG=<registry>/nrp-gpu-bench:latest       # e.g. gitlab-registry.nrp-nautilus.io/<you>/nrp-gpu-bench:latest
```

Everything is namespaced. Either bake `$NS` into your context (Step 2) or pass
`-n $NS` on every command. The runbook assumes you set the context.

---

## Step 0 — Prerequisites (one time)

On your **workstation**:

- `kubectl` — installed and pointed at Nautilus. Get your kubeconfig from the
  NRP User Portal (Get Config), drop it in `~/.kube/config`, and verify:
  ```bash
  kubectl config current-context
  kubectl auth can-i list nodes          # should say "yes" (your user has node read)
  ```
- `python3` — 3.9+ (standard library only; nothing to pip install for the tools).
- A container builder — `docker` with `buildx`, or `podman`.
- Push access to a registry. The NRP GitLab registry
  (`gitlab-registry.nrp-nautilus.io/<you>/...`) is the natural choice; Docker Hub
  works too.
- **Claude Code CLI** (optional but handy — see the appendix): `claude` in the
  repo directory becomes an assistant that can run kubectl, read logs, and
  interpret results for you.

Clone this repo onto your workstation (use the URL of wherever you're reading it):

```bash
git clone <this-repo-url>
cd K8s-GPU-Scraper-bench     # or whatever the repo directory is named
```

---

## Step 1 — Confirm the cluster basics

```bash
# What storage class exists? The manifests assume rook-ceph-block.
kubectl get storageclass
# If rook-ceph-block is NOT listed, edit the storageClassName in the three
# k8s/*.yaml PVCs to a class that IS listed before applying.
```

---

## Step 2 — Point kubectl at your namespace

```bash
kubectl config set-context --current --namespace=$NS
kubectl get pods            # should return "No resources found" cleanly
```

(SLURM analogy: this is like defaulting your `--account`/partition so you don't
pass it on every `sbatch`.)

---

## Step 3 — Set a real Grafana password

The committed secret is a `CHANGE-ME` placeholder. Override it **before** you
apply, so it never lands on the cluster:

```bash
kubectl create secret generic grafana-admin \
  --from-literal=admin-password='<pick-a-strong-password>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## Step 4 — Deploy the monitoring stack

```bash
# Optional: preview exactly what will be created (no changes made).
kubectl kustomize k8s/ | less

# Apply it.
kubectl apply -k k8s/

# Watch the three Deployments come up. Ctrl-C when all three are Running.
kubectl get pods -w
#   Pending           -> waiting for a node or PVC to bind
#   ContainerCreating -> pulling images, mounting volumes
#   Running           -> serving
```

Sanity checks:

```bash
kubectl get pvc                 # prometheus-data, pushgateway-data, grafana-data -> Bound
kubectl get svc                 # prometheus:9090, pushgateway:9091, grafana:3000
```

Open Grafana (leave this running in its own terminal):

```bash
kubectl port-forward svc/grafana 3000:3000
# Browse http://localhost:3000  — login: admin / <your password from Step 3>
# The "NRP GPU Comparison" dashboard is pre-provisioned (empty until you run a benchmark).
```

---

## Step 5 — Discover the GPU flavors

This runs **on your laptop** with your node-read credentials — no cluster
workload involved.

```bash
# Human-readable catalog:
kubectl get nodes -o json | python3 inventory/gpu_inventory.py --format md

# Machine-readable (this is what submit.py consumes):
kubectl get nodes -o json | python3 inventory/gpu_inventory.py --format json -o catalog.json
```

Read the `Restricted` column: flavors behind a reservation/system/issue taint
are shown but will be **skipped** by `submit.py` (you can't tolerate those
without admin authorization).

---

## Step 6 — Build and push the benchmark image

The benchmark code only reaches the cluster as an image. Build **multi-arch** so
the arm64 GH200/Grace nodes work too:

```bash
# Log in to your registry first (e.g. NRP GitLab registry):
#   docker login gitlab-registry.nrp-nautilus.io

docker buildx build --platform linux/amd64,linux/arm64 \
  -t $IMG --push bench/
```

No buildx? For an amd64-only first pass:

```bash
docker build -t $IMG bench/ && docker push $IMG
# (amd64-only image cannot run on GH200; use --no-arch-tolerate in Step 7.)
```

> **Note:** the Dockerfile pins `nvidia-ml-py` and `prometheus_client` to guessed
> versions and was never built in this project. If pip errors with "no matching
> distribution," bump those two pins in `bench/Dockerfile` and rebuild.

---

## Step 7 — Run benchmarks (the tier workflow)

`submit.py` reads the flavor catalog and emits one Job per flavor. You control
**which flavors** (`--only`, `--min-gpus`, `--max-jobs`) and **which tiers**
(`--tiers`). Pattern:

```bash
kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py --image $IMG <flags> \
  | kubectl apply -f -
```

### The three tiers — what each measures and when to run it

| Tier | `--tiers` value | Measures | GPU load | Runtime | Run it when |
|---|---|---|---|---|---|
| **0 — inventory** | `inventory` | device name, compute capability, VRAM, SM count, driver/CUDA, power limit, PCIe gen/width | ~0% | seconds | You want the card's *spec sheet as measured*. Cheap, but near-zero GPU util — never spray many of these at once (ban risk). |
| **1 — micro** | `micro` | GEMM TFLOPS (fp32/tf32/bf16/fp16), STREAM memory bandwidth GB/s, PCIe H2D/D2H GB/s | high during GEMM | ~1–3 min | You want raw peak numbers to compare architectures. |
| **2 — workload** | `workload` | a real Transformer training step → **tokens/sec** | high, sustained | a few min | You want the number that reflects actual research/LLM work — the most representative and the most decision-relevant. |

Tiers are additive — always lead with `inventory` so every push carries the
spec-sheet facts (`nrpbench_gpu_info`).

### 7a — First run: ONE flavor, cheap tiers

Start here. Prove the whole pipeline end to end before scaling.

```bash
kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py --image $IMG --only 3090 --tiers inventory,micro \
  | kubectl apply -f -

# Follow it. Expect: pre-scrape sleep (~15s), benchmarks, "results pushed", post-scrape sleep (~20s).
kubectl logs -l app=nrp-gpu-bench -f
```

Give Prometheus ~a minute after `results pushed`, then look at Grafana. The
comparison matrix should show one row; the live-telemetry panels show the run's
utilization/power/temp curve.

### 7b — Add the full workload tier (same flavor)

Jobs auto-clean after 24h (`ttlSecondsAfterFinished`), but to re-run *now* you
must delete the finished Job first (names are deterministic per flavor):

```bash
kubectl delete job -l app=nrp-gpu-bench            # clear the previous run(s)

kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py --image $IMG --only 3090 --tiers inventory,micro,workload \
  | kubectl apply -f -
kubectl logs -l app=nrp-gpu-bench -f
```

### 7c — Several flavors at once (mind the queue)

There is **no fair queue** on Nautilus — a bulk apply blocks everyone.
`submit.py` caps output at `--max-jobs 5` by default. Widen deliberately:

```bash
# Three specific flavors, full tiers:
kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py --image $IMG \
      --only 3090 --only a40 --only v100 \
      --tiers inventory,micro,workload --max-jobs 5 \
  | kubectl apply -f -
```

### 7d — Gated GPUs (A100 / H100 / H200 / GH200)

These sit behind a per-namespace quota that defaults to zero. Use the sanctioned
preemptible path:

```bash
kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py --image $IMG --only a100 --opportunistic \
      --tiers inventory,micro \
  | kubectl apply -f -
```

- `--opportunistic` = quota doesn't apply, but the pod can be **preempted** at
  any time (it's a Job, so it just retries). This is documented and allowed.
- For a *reserved* (non-preemptible) A100, fill out the A100 access form instead.
- GH200/H100/H200 have **no form** — opportunistic is the only user path.
- GH200 also needs the **arm64 image** from Step 6 (multi-arch build). If your
  image is amd64-only, add `--no-arch-tolerate` to skip arm64 flavors.

### 7e — Variance: run a flavor several times

Cooling and PCIe topology differ between nodes of the same flavor. To sample
variance, delete + resubmit the same flavor a few times (the scheduler may place
it on different nodes); the dashboard's `max by (product)` keeps the best:

```bash
for i in 1 2 3; do
  kubectl delete job -l app=nrp-gpu-bench 2>/dev/null || true
  kubectl get nodes -o json \
    | python3 inventory/gpu_inventory.py --format json \
    | python3 bench/submit.py --image $IMG --only 3090 --tiers micro \
    | kubectl apply -f -
  kubectl wait --for=condition=complete job -l app=nrp-gpu-bench --timeout=600s
done
```

### Useful submit.py flags

| Flag | Effect |
|---|---|
| `--only SUBSTR` | keep flavors whose product/resource contains SUBSTR (repeatable) |
| `--tiers a,b,c` | which tiers to run (`inventory,micro,workload`) |
| `--max-jobs N` | cap emitted Jobs (default 5; `0` = unlimited, prints a warning) |
| `--opportunistic` | set the preemptible priority class for gated GPUs |
| `--no-arch-tolerate` | skip arm64 flavors (use when your image is amd64-only) |
| `--min-gpus N` | skip flavors with fewer than N schedulable GPUs |
| `--gpus N` | GPUs per Job (1/2/4/8; default 1 — keep it busy) |
| `--include-restricted` | also emit restricted flavors (they'll likely stay Pending) |

Preview without applying — just drop the `| kubectl apply -f -`:

```bash
... | python3 bench/submit.py --image $IMG --only 3090 --tiers micro
```

---

## Step 8 — Read the results in Grafana

With the port-forward from Step 4 running, open the **NRP GPU Comparison**
dashboard. Read it in this order:

1. **Memory bandwidth** — most real work is memory-bound; read this before TFLOPS.
2. **Comparison matrix** — VRAM, FP32/BF16 TFLOPS, bandwidth, tokens/sec, TDP,
   and the computed **TFLOPS/W** (the purchasing column, gradient-colored).
3. **Temperature + SM clock** — a clock that sags as temp rises = thermal
   throttling; that run is a floor, not the card's ceiling.
4. **Result freshness** — re-run anything showing as stale.

Switch the **GEMM dtype** variable (top of the dashboard) to compare fp32 vs bf16.

Prometheus directly (optional):

```bash
kubectl port-forward svc/prometheus 9090:9090   # http://localhost:9090
# /targets  -> confirm your bench pod was discovered and scraped
# try a query:  max by (product) (nrpbench_gemm_tflops{dtype="bf16"})
```

---

## Step 9 — Debugging loop (something will be wrong)

```bash
kubectl get pods -l app=nrp-gpu-bench            # STATUS tells you a lot
kubectl describe job/<job-name>                  # Job-level conditions
kubectl describe pod/<pod-name>                  # read the Events section (bottom) FIRST
kubectl get events --sort-by=.lastTimestamp      # recent cluster events, newest last
kubectl logs <pod-name> --previous               # logs from a crashed prior attempt
```

Common causes:

| Symptom | Likely cause | Fix |
|---|---|---|
| Pod `Pending` forever | quota-gated GPU, or a taint you didn't tolerate | `--opportunistic`; check the flavor isn't `restricted` |
| `Pending`, "Insufficient nvidia.com/..." | no schedulable capacity right now | try another flavor / later |
| Pod errors on arm64 node | amd64-only image | rebuild multi-arch, or `--no-arch-tolerate` |
| Dashboard empty after a run | scraped too late, or image wrong arch | confirm `/targets`; the post-scrape delay covers timing |
| All flavors show identical numbers | `honor_labels` missing (shouldn't happen here) | verify `k8s/10-prometheus.yaml` |

---

## Step 10 — Cleanup / re-run later

```bash
# Remove finished benchmark Jobs (results persist in the Pushgateway):
kubectl delete job -l app=nrp-gpu-bench

# Evict one flavor's stored result from the Pushgateway (P=product, N=node).
# Port-forward from your workstation and curl it (the pushgateway image is
# minimal, so drive the API from outside rather than kubectl exec):
kubectl port-forward svc/pushgateway 9091:9091 &
curl -X DELETE "http://localhost:9091/metrics/job/nrp_gpu_bench/product/<P>/node/<N>"
kill %1        # stop the port-forward

# Tear the whole stack down (PVCs and their data SURVIVE):
kubectl delete -k k8s/
# Re-applying later restores everything from the PVCs:
kubectl apply -k k8s/
```

> **2-week purge:** the monitoring Deployments are auto-deleted after two weeks
> unless your namespace has a permanent-service exception (ask Nautilus Support).
> Because the PVCs survive, `kubectl apply -k k8s/` brings it all back with
> history intact. To keep it running continuously, request the exception.

---

## Appendix A — Using the Claude Code CLI as your co-pilot

Run `claude` from inside the repo on your workstation. It has your kubeconfig and
the repo context, so you can hand it the tedious parts:

- **Deploy & watch:** "apply the k8s stack and tell me when all three pods are Running."
- **Generate a submit command:** "build me a submit.py command to benchmark the
  A100 and L40 with the workload tier, opportunistic."
- **Debug:** "my 3090 job is stuck Pending — describe the pod and tell me why."
- **Interpret:** "pull the bf16 TFLOPS and TDP for every flavor from Prometheus
  and rank them by TFLOPS/W."
- **Refresh the inventory ConfigMap** (for the CronJob):
  "recreate the gpu-inventory-script configmap from the current scraper."

It runs the same `kubectl`/`python3` commands you would — you approve each one.
Nothing here needs Claude; it just removes the copy-paste.

---

## Appendix B — Quick command index

```bash
# Deploy
kubectl apply -k k8s/
kubectl port-forward svc/grafana 3000:3000

# Inventory
kubectl get nodes -o json | python3 inventory/gpu_inventory.py --format md

# Benchmark one flavor, cheap tiers
kubectl get nodes -o json \
  | python3 inventory/gpu_inventory.py --format json \
  | python3 bench/submit.py --image $IMG --only 3090 --tiers inventory,micro \
  | kubectl apply -f -

# Watch / debug
kubectl logs -l app=nrp-gpu-bench -f
kubectl describe pod/<pod>
kubectl get events --sort-by=.lastTimestamp

# Cleanup
kubectl delete job -l app=nrp-gpu-bench
```
