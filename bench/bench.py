#!/usr/bin/env python3
"""GPU benchmark harness for the NRP "Nautilus" cluster.

One run measures one GPU and does three things (BUILD-SPEC §4 Phase 3):

  1. Live telemetry  -- an NVML sampler thread exposes utilization / power /
     temp / SM clock / memory as Prometheus gauges on :9400. Prometheus scrapes
     these off the running pod every 5s; they are meaningful only during the run
     and correctly disappear when the pod dies.
  2. Benchmarks      -- Tier 0 inventory, Tier 1 microbenchmarks (GEMM, memory
     bandwidth, PCIe), Tier 2 a real Transformer training step. Each is
     individually guarded: an unsupported dtype records a skip and the run
     continues, it never aborts the whole run.
  3. Durable results -- TFLOPS / GB-s / tokens-per-sec are pushed once to the
     Pushgateway with grouping key (product, node), so they survive the pod.

Two registries on purpose (§2.1): live gauges on the default registry (scraped),
result gauges on a fresh CollectorRegistry (pushed). See push_results().

The program ALWAYS terminates on its own (§2.4): bounded pre/post-scrape sleeps
around work that exits, and a *daemon* sampler thread that dies with the process.
There is no `sleep infinity` anywhere -- that is bannable here.

Env contract:
  PUSHGATEWAY       Pushgateway base URL      (default http://pushgateway:9091)
  NODE_NAME         node name (downward API)  (default: hostname)
  GPU_PRODUCT       GPU product label         (default: unknown)
  BENCH_TIERS       comma list to run         (default inventory,micro,workload)
  METRICS_PORT      live exporter port        (default 9400)
  RESULTS_JSON      also write results here    (optional)
  PRESCRAPE_DELAY   idle secs before load     (default 15)  -- see F3 note below
  POSTSCRAPE_DELAY  idle secs after push      (default 20)

Policy note (F3, POLICY.md): a GPU-requesting Job must stay >=40% utilized. The
pre/post-scrape windows are one-time and dwarfed by real compute, so a normal
run is fine -- but do NOT spray many `inventory`-only (near-zero-util) GPU Jobs
at once, or >4 concurrent low-util pods can get the namespace banned.
"""

import functools
import json
import os
import socket
import sys
import threading
import time

# --- Soft dependencies -------------------------------------------------------
# torch and pynvml may be missing (old card, or a no-GPU test box). Import them
# softly so the module still imports; guarded benchmarks then record a skip
# instead of raising. prometheus_client is also soft so the module imports
# without it, but the exporter and push_results genuinely need it.
try:
    import torch
except Exception:                                       # pragma: no cover
    torch = None

try:
    import pynvml
except Exception:                                       # pragma: no cover
    pynvml = None

try:
    from prometheus_client import (CollectorRegistry, Gauge, push_to_gateway,
                                   start_http_server)
except Exception:                                       # pragma: no cover
    CollectorRegistry = Gauge = push_to_gateway = start_http_server = None


# --- Config from the environment ---------------------------------------------
PUSHGATEWAY = os.environ.get("PUSHGATEWAY", "http://pushgateway:9091")
NODE_NAME = os.environ.get("NODE_NAME") or socket.gethostname()
GPU_PRODUCT = os.environ.get("GPU_PRODUCT", "unknown")
BENCH_TIERS = os.environ.get("BENCH_TIERS", "inventory,micro,workload")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9400"))
RESULTS_JSON = os.environ.get("RESULTS_JSON", "")
PRESCRAPE_DELAY = float(os.environ.get("PRESCRAPE_DELAY", "15"))
POSTSCRAPE_DELAY = float(os.environ.get("POSTSCRAPE_DELAY", "20"))

PUSH_JOB = "nrp_gpu_bench"   # grouping job name (§2.3)

# Collected results, pushed at the end and optionally dumped to RESULTS_JSON.
RESULTS = {"skipped": []}


# --- Live telemetry gauges (default registry, scraped on :9400) --------------
# Created only if prometheus_client is importable, so the module still imports
# on a bare test box.
if Gauge is not None:
    _LIVE = ["gpu", "product"]
    GPU_UTIL = Gauge("nrpbench_gpu_utilization_percent",
                     "GPU utilization percent", _LIVE)
    GPU_MEM = Gauge("nrpbench_gpu_memory_used_bytes",
                    "GPU memory used, bytes", _LIVE)
    GPU_POWER = Gauge("nrpbench_gpu_power_watts", "GPU power draw, watts", _LIVE)
    GPU_TEMP = Gauge("nrpbench_gpu_temperature_celsius",
                     "GPU temperature, C", _LIVE)
    GPU_SMCLK = Gauge("nrpbench_gpu_sm_clock_mhz", "SM clock, MHz", _LIVE)
    # 1 while the named phase runs, 0 otherwise -- lets the Grafana timeline be
    # read against the live telemetry.
    PHASE = Gauge("nrpbench_phase", "1 while the named benchmark phase runs",
                  ["phase"])
else:                                                   # pragma: no cover
    GPU_UTIL = GPU_MEM = GPU_POWER = GPU_TEMP = GPU_SMCLK = PHASE = None

_PHASES = ("prescrape", "inventory", "gemm", "membw", "pcie", "workload",
           "push", "postscrape", "done")


def set_phase(name):
    """Mark the current phase (1) and clear the others (0). Never fatal."""
    if PHASE is None:
        return
    for phase in _PHASES:
        try:
            PHASE.labels(phase=phase).set(1.0 if phase == name else 0.0)
        except Exception:                               # pragma: no cover
            pass


# --- Guard decorator (§2.7) --------------------------------------------------
def guarded(name):
    """Catch anything, log a skip, record it, return None -- never abort a run."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                label = name
                if args:
                    label = f"{name}/{args[0]}"         # e.g. gemm/fp32
                print(f"[skip] {label}: {exc}", file=sys.stderr)
                RESULTS["skipped"].append({"name": label, "error": str(exc)})
                return None
        return wrapper
    return decorate


# --- Live NVML sampler -------------------------------------------------------
def _nvml_str(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _sample_loop(product):
    """Sample NVML every 2s. Every read is wrapped so a hiccup drops one
    sample rather than crashing the run."""
    if pynvml is None:
        print("[warn] pynvml missing; no live GPU telemetry", file=sys.stderr)
        return
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        print(f"[warn] NVML init failed; no live telemetry: {exc}",
              file=sys.stderr)
        return
    gpu = "0"
    readers = (
        (GPU_UTIL, lambda h: pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
        (GPU_MEM, lambda h: pynvml.nvmlDeviceGetMemoryInfo(h).used),
        (GPU_POWER, lambda h: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0),
        (GPU_TEMP, lambda h: pynvml.nvmlDeviceGetTemperature(
            h, pynvml.NVML_TEMPERATURE_GPU)),
        (GPU_SMCLK, lambda h: pynvml.nvmlDeviceGetClockInfo(
            h, pynvml.NVML_CLOCK_SM)),
    )
    while True:
        for gauge, read in readers:
            try:
                gauge.labels(gpu=gpu, product=product).set(read(handle))
            except Exception:
                pass                                    # drop one sample, live on
        time.sleep(2)


def start_exporter(product, port):
    """Start the :9400 metrics server and the sampler thread. Never fatal."""
    if start_http_server is None:
        print("[warn] prometheus_client missing; no live exporter",
              file=sys.stderr)
        return False
    try:
        start_http_server(port)
    except Exception as exc:                            # pragma: no cover
        print(f"[warn] exporter failed to start on :{port}: {exc}",
              file=sys.stderr)
        return False
    threading.Thread(target=_sample_loop, args=(product,),
                     daemon=True).start()
    return True


# --- CUDA-event timing (§4 Phase 3c) -----------------------------------------
def time_cuda(fn, warmup=5, iters=20):
    """Seconds per iteration, timed with CUDA events. Kernel launches are
    asynchronous, so wall-clock timing is wrong -- warm up (JIT / autotune /
    clock ramp), then bracket the work with events and synchronize."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters     # ms -> s, per iter


# --- helpers -----------------------------------------------------------------
_DTYPE_BYTES = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2}


def _torch_dtype(name):
    return {"fp32": torch.float32, "tf32": torch.float32,
            "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _cc_supports(dtype, cc):
    """Dtype availability gated on compute capability (§4 Phase 3e)."""
    if dtype == "fp32":
        return True
    if dtype in ("tf32", "bf16"):
        return cc >= 8.0
    if dtype == "fp16":
        return cc >= 7.0
    return False


def device_compute_capability():
    """Compute capability as a float (8.6), or None if no CUDA / no torch."""
    try:
        props = torch.cuda.get_device_properties(0)
        return float(f"{props.major}.{props.minor}")
    except Exception:
        return None


def _square_n(free_bytes, dtype_bytes, frac=0.25):
    """Largest N for three NxN matrices in ~frac of free VRAM, capped at 8192,
    rounded down to a multiple of 256."""
    target = free_bytes * frac
    n = int((target / (3 * dtype_bytes)) ** 0.5)
    n = min(n, 8192)
    n -= n % 256
    return max(n, 256)


# --- Tier 0: inventory -------------------------------------------------------
@guarded("inventory")
def collect_inventory():
    props = torch.cuda.get_device_properties(0)
    inv = {
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_total_bytes": int(props.total_memory),
        "sm_count": int(props.multi_processor_count),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    # NVML extras are best-effort -- a partial failure still returns the torch
    # facts above.
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        inv["driver_version"] = _nvml_str(pynvml.nvmlSystemGetDriverVersion())
        inv["power_limit_watts"] = (
            pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
        inv["pcie_gen"] = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        inv["pcie_width"] = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
    except Exception as exc:
        print(f"[warn] NVML inventory partial: {exc}", file=sys.stderr)
    return inv


# --- Tier 1: microbenchmarks -------------------------------------------------
@guarded("gemm")
def gemm_tflops(dtype):
    dev = torch.device("cuda")
    free, _ = torch.cuda.mem_get_info()
    n = _square_n(free, _DTYPE_BYTES[dtype])
    td = _torch_dtype(dtype)
    a = torch.randn(n, n, device=dev, dtype=td)
    b = torch.randn(n, n, device=dev, dtype=td)
    c = torch.empty(n, n, device=dev, dtype=td)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # fp32 must disable TF32 or Ampere+ silently runs TF32 and the two
        # measurements come out identical (§2 load-bearing). Reset in finally.
        torch.backends.cuda.matmul.allow_tf32 = (dtype == "tf32")
        seconds = time_cuda(lambda: torch.matmul(a, b, out=c))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return 2 * (n ** 3) / seconds / 1e12


@guarded("membw")
def mem_bandwidth_gbps():
    """STREAM triad a = b + s*c. Three float32 arrays touched per element."""
    dev = torch.device("cuda")
    free, _ = torch.cuda.mem_get_info()
    n = int(free * 0.25 / (3 * 4))
    n -= n % 256
    n = max(n, 256)
    b = torch.randn(n, device=dev, dtype=torch.float32)
    c = torch.randn(n, device=dev, dtype=torch.float32)
    a = torch.empty(n, device=dev, dtype=torch.float32)
    seconds = time_cuda(lambda: torch.add(b, c, alpha=2.0, out=a))
    return (3 * n * 4) / seconds / 1e9


@guarded("pcie")
def pcie_bandwidth_gbps():
    """Pinned-host <-> device copies, timed with CUDA events. A downgraded or
    narrow link shows up here; Gen4 x16 tops out around 25 GB/s."""
    dev = torch.device("cuda")
    n = 64 * 1024 * 1024                                 # 64M float32 = 256 MiB
    host = torch.randn(n, dtype=torch.float32).pin_memory()
    devt = torch.empty(n, dtype=torch.float32, device=dev)
    nbytes = n * 4
    t_h2d = time_cuda(lambda: devt.copy_(host, non_blocking=True))
    t_d2h = time_cuda(lambda: host.copy_(devt, non_blocking=True))
    return {"h2d": nbytes / t_h2d / 1e9, "d2h": nbytes / t_d2h / 1e9}


# --- Tier 2: workload --------------------------------------------------------
@guarded("workload")
def transformer_tokens_per_sec():
    dev = torch.device("cuda")
    cc = device_compute_capability() or 0.0
    dtype = torch.bfloat16 if cc >= 8.0 else torch.float16
    gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    batch = 16 if gib >= 40 else 8 if gib >= 20 else 4
    seq, d_model, nhead = 1024, 2048, 16
    # Construct then cast -- safer across torch versions than the dtype= kwarg.
    layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dropout=0.0, batch_first=True)
    layer = layer.to(dev).to(dtype)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    x = torch.randn(batch, seq, d_model, device=dev, dtype=dtype)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = layer(x).float().mean()
        loss.backward()
        opt.step()

    seconds = time_cuda(step, warmup=3, iters=10)
    return {"tokens_per_sec": batch * seq / seconds,
            "dtype": "bf16" if dtype == torch.bfloat16 else "fp16",
            "batch": batch}


# --- Tier runners ------------------------------------------------------------
def run_inventory():
    inv = collect_inventory()
    if inv:
        RESULTS["inventory"] = inv


def run_micro():
    cc = device_compute_capability()
    if cc is None:
        RESULTS["skipped"].append({"name": "micro", "error": "no CUDA device"})
        return
    set_phase("gemm")
    gemm = {}
    for dtype in ("fp32", "tf32", "bf16", "fp16"):
        if not _cc_supports(dtype, cc):
            RESULTS["skipped"].append(
                {"name": f"gemm/{dtype}",
                 "error": f"compute capability {cc} too low for {dtype}"})
            continue
        val = gemm_tflops(dtype)
        if val is not None:
            gemm[dtype] = val
    if gemm:
        RESULTS["gemm"] = gemm
    set_phase("membw")
    mb = mem_bandwidth_gbps()
    if mb is not None:
        RESULTS["mem_bandwidth_gbps"] = mb
    set_phase("pcie")
    pcie = pcie_bandwidth_gbps()
    if pcie is not None:
        RESULTS["pcie"] = pcie


def run_workload():
    wl = transformer_tokens_per_sec()
    if wl is not None:
        RESULTS["workload"] = wl


# --- Push durable results (§2.3) ---------------------------------------------
def push_results(results, gateway=None, product=None, node=None):
    """Build a FRESH registry, populate result gauges, push with grouping key
    (product, node). Raising URLError here means every label set was valid and
    only the network failed; a ValueError would mean a label mismatch."""
    if CollectorRegistry is None:                       # pragma: no cover
        raise RuntimeError("prometheus_client not installed")
    gateway = gateway or PUSHGATEWAY
    product = product or GPU_PRODUCT
    node = node or NODE_NAME
    reg = CollectorRegistry()

    # Info metric: value always 1, static facts carried as labels (the standard
    # Prometheus "info" pattern).
    inv = results.get("inventory") or {}
    info = Gauge("nrpbench_gpu_info", "Static GPU facts; value is always 1",
                 ["device", "compute_capability", "driver", "cuda", "torch"],
                 registry=reg)
    info.labels(device=str(inv.get("device_name", "")),
                compute_capability=str(inv.get("compute_capability", "")),
                driver=str(inv.get("driver_version", "")),
                cuda=str(inv.get("cuda_version", "")),
                torch=str(inv.get("torch_version", ""))).set(1)

    if inv.get("vram_total_bytes") is not None:
        Gauge("nrpbench_gpu_vram_total_bytes", "Total VRAM, bytes",
              registry=reg).set(inv["vram_total_bytes"])
    if inv.get("power_limit_watts") is not None:
        Gauge("nrpbench_gpu_power_limit_watts", "Enforced power limit, watts",
              registry=reg).set(inv["power_limit_watts"])

    gemm = results.get("gemm") or {}
    if gemm:
        g = Gauge("nrpbench_gemm_tflops", "GEMM throughput, TFLOPS",
                  ["dtype"], registry=reg)
        for dtype, val in gemm.items():
            if val is not None:
                g.labels(dtype=dtype).set(val)

    if results.get("mem_bandwidth_gbps") is not None:
        Gauge("nrpbench_memory_bandwidth_gbps", "STREAM triad bandwidth, GB/s",
              registry=reg).set(results["mem_bandwidth_gbps"])

    pcie = results.get("pcie") or {}
    if pcie:
        p = Gauge("nrpbench_pcie_bandwidth_gbps", "PCIe copy bandwidth, GB/s",
                  ["direction"], registry=reg)
        for direction, val in pcie.items():
            if val is not None:
                p.labels(direction=direction).set(val)

    wl = results.get("workload") or {}
    if wl.get("tokens_per_sec") is not None:
        Gauge("nrpbench_workload_tokens_per_second",
              "Transformer training throughput, tokens/sec",
              registry=reg).set(wl["tokens_per_sec"])

    # Freshness: lets the dashboard flag stale flavors.
    Gauge("nrpbench_run_timestamp_seconds", "Unix time this run finished",
          registry=reg).set(time.time())

    push_to_gateway(gateway, job=PUSH_JOB,
                    grouping_key={"product": product, "node": node},
                    registry=reg)


# --- Orchestration -----------------------------------------------------------
def main():
    print(f"nrpbench: product={GPU_PRODUCT} node={NODE_NAME} "
          f"tiers={BENCH_TIERS}", file=sys.stderr)
    start_exporter(GPU_PRODUCT, METRICS_PORT)

    # Let Prometheus discover and scrape the pod before load starts.
    set_phase("prescrape")
    time.sleep(PRESCRAPE_DELAY)

    tiers = [t.strip() for t in BENCH_TIERS.split(",") if t.strip()]
    if "inventory" in tiers:
        set_phase("inventory")
        run_inventory()
    if "micro" in tiers:
        run_micro()
    if "workload" in tiers:
        set_phase("workload")
        run_workload()

    set_phase("push")
    if RESULTS_JSON:
        try:
            with open(RESULTS_JSON, "w") as fh:
                json.dump(RESULTS, fh, indent=2)
        except Exception as exc:
            print(f"[warn] could not write {RESULTS_JSON}: {exc}",
                  file=sys.stderr)
    try:
        push_results(RESULTS)
        print("nrpbench: results pushed", file=sys.stderr)
    except Exception as exc:
        # A dead Pushgateway loses this run's results but must not wedge the
        # Job into a retry storm; log and exit cleanly.
        print(f"[error] push failed: {exc}", file=sys.stderr)

    # Let the final live scrape land before we exit.
    set_phase("postscrape")
    time.sleep(POSTSCRAPE_DELAY)
    set_phase("done")
    print("nrpbench: done", file=sys.stderr)


if __name__ == "__main__":
    main()
