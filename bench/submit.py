#!/usr/bin/env python3
"""Turn a flavor catalog (gpu_inventory.py --format json) into benchmark Jobs.

Reads the catalog on stdin (or --inventory FILE) and emits one Kubernetes Job
per flavor to stdout as a multi-document stream. Output is JSON documents
separated by `---`; JSON is a subset of YAML, so `kubectl apply -f -` and
`yaml.safe_load_all` both accept it and this script needs no third-party deps.

    gpu_inventory.py --format json < nodes.json \
      | submit.py --image <registry>/nrp-gpu-bench:latest --tiers inventory,micro \
      | kubectl apply -f -

The core branch (BUILD-SPEC §2.6):
  * special resource name (e.g. nvidia.com/rtx8000) -> request it BY NAME, no
    nodeSelector; requesting the named resource already pins the hardware.
  * generic nvidia.com/gpu -> request nvidia.com/gpu PLUS a
    nodeSelector {nvidia.com/gpu.product: <product>} to pin the model.

Policy (POLICY.md), because NRP docs win over the spec:
  * F1 -- tolerations are the flavor's derived `tolerations` only (the
    nautilus.io/hardware tier). We never invent one. Flavors marked `restricted`
    (reservation / system / issue / science-dmz taints) are SKIPPED by default;
    tolerating a taint you are not authorized for is itself a violation.
  * F2 -- there is no fair queue on Nautilus; a bulk apply blocks everyone.
    --max-jobs caps how many Jobs are emitted (default 5). Raise it deliberately.
  * F5 -- --opportunistic sets priorityClassName: opportunistic. This is the
    NRP-sanctioned way to reach A100/H100/H200/GH200 without a reservation: the
    per-namespace quota does not apply to opportunistic-tier pods, in exchange
    for preemption at any time. It is a documented feature, NOT a circumvention.
    Distinction: the A100 access-request form grants *reserved* (non-preemptible)
    quota; H100/H200/GH200 have no form, so opportunistic is the only user path.
    Only these priority classes pass admission: unset, armada-default,
    owner-no-preempt, opportunistic, opportunistic2 -- we never emit another.
"""

import argparse
import json
import re
import sys

MAX_NAME = 52          # DNS-1123 label budget for the Job name
METRICS_PORT = 9400
ALLOWED_PRIORITY = {"opportunistic", "opportunistic2"}


def dns1123(text, maxlen):
    """Lowercase, alphanumeric + single dashes, trimmed, <= maxlen."""
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    text = re.sub(r"-+", "-", text).strip("-")
    if len(text) > maxlen:
        text = text[:maxlen].strip("-")
    return text or "x"


def job_namer():
    """Return a function that yields unique DNS-1123 Job names."""
    seen = set()

    def name_for(product):
        base = "nrpbench-" + dns1123(product, MAX_NAME - len("nrpbench-"))
        name = base
        i = 1
        while name in seen:
            i += 1
            suffix = f"-{i}"
            name = base[:MAX_NAME - len(suffix)] + suffix
        seen.add(name)
        return name

    return name_for


def keep_flavor(flavor, args):
    """Filter decision + reason. Returns (keep: bool, skip_reason: str|None)."""
    product = flavor.get("product", "")
    resource = flavor.get("resource", "")
    if flavor.get("restricted") and not args.include_restricted:
        reasons = ",".join(t.get("reason", "?")
                           for t in flavor.get("restricted_taints", []))
        return False, f"restricted ({reasons}); needs admin authorization"
    if flavor.get("gpus_schedulable", 0) < args.min_gpus:
        return False, (f"{flavor.get('gpus_schedulable', 0)} schedulable GPUs "
                       f"< --min-gpus {args.min_gpus}")
    if flavor.get("arch") == "arm64" and not args.arch_tolerate:
        return False, "arm64 flavor and --no-arch-tolerate (need arm64 image)"
    if args.only:
        hay = f"{product} {resource}".lower()
        if not any(sub.lower() in hay for sub in args.only):
            return False, f"no --only filter matched"
    return True, None


def build_job(flavor, name, args):
    product = flavor.get("product", "unknown")
    resource = flavor.get("resource", "nvidia.com/gpu")
    gpu_count = str(args.gpus)

    labels = {"app": "nrp-gpu-bench", "gpu-product": product}

    env = [
        {"name": "NODE_NAME",
         "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "GPU_PRODUCT", "value": product},
        {"name": "BENCH_TIERS", "value": args.tiers},
        {"name": "PUSHGATEWAY", "value": args.pushgateway},
        {"name": "METRICS_PORT", "value": str(METRICS_PORT)},
    ]

    requests = {"cpu": args.cpu_request, "memory": args.mem_request,
                "ephemeral-storage": args.eph_request, resource: gpu_count}
    limits = {"cpu": args.cpu_limit, "memory": args.mem_limit,
              "ephemeral-storage": args.eph_limit, resource: gpu_count}

    pod_spec = {
        "restartPolicy": "Never",
        "containers": [{
            "name": "bench",
            "image": args.image,
            "env": env,
            "ports": [{"name": "metrics", "containerPort": METRICS_PORT}],
            "resources": {"requests": requests, "limits": limits},
        }],
    }

    # §2.6 branch: named resource pins hardware; generic gpu needs a selector.
    if flavor.get("needs_node_selector"):
        pod_spec["nodeSelector"] = {"nvidia.com/gpu.product": product}

    # F1: emit ONLY the flavor's derived (hardware-tier) tolerations.
    if flavor.get("tolerations"):
        pod_spec["tolerations"] = flavor["tolerations"]

    # F5: opportunistic is the only priority class we ever set, and only on ask.
    if args.opportunistic:
        pod_spec["priorityClassName"] = "opportunistic"

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": labels,
                    "annotations": {
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port": str(METRICS_PORT),
                    },
                },
                "spec": pod_spec,
            },
        },
    }


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Emit benchmark Jobs from a GPU flavor catalog.")
    p.add_argument("--image", required=True,
                   help="benchmark container image (required)")
    p.add_argument("--inventory", help="flavor catalog JSON (default: stdin)")
    p.add_argument("--tiers", default="inventory,micro,workload",
                   help="BENCH_TIERS for the pods")
    p.add_argument("--only", action="append", default=[], metavar="SUBSTR",
                   help="keep flavors whose product/resource contains SUBSTR "
                        "(repeatable)")
    p.add_argument("--min-gpus", type=int, default=1,
                   help="skip flavors with fewer schedulable GPUs (default 1)")
    p.add_argument("--max-jobs", type=int, default=5,
                   help="cap emitted Jobs -- there is no fair queue (F2). "
                        "0 = unlimited (prints a loud warning)")
    p.add_argument("--opportunistic", action="store_true",
                   help="set priorityClassName: opportunistic (sanctioned "
                        "no-reservation path to gated GPUs; preemptible)")
    p.add_argument("--arch-tolerate", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="target arm64 (Grace/GH200) flavors; needs a multi-arch "
                        "image. --no-arch-tolerate skips them (default: on)")
    p.add_argument("--include-restricted", action="store_true",
                   help="also emit Jobs for restricted flavors (they will "
                        "likely stay Pending without admin authorization)")
    p.add_argument("--gpus", type=int, default=1,
                   help="GPUs requested per Job (default 1; keep it busy, F3)")
    p.add_argument("--pushgateway", default="http://pushgateway:9091")
    p.add_argument("--cpu-request", default="2")
    p.add_argument("--cpu-limit", default="4")
    p.add_argument("--mem-request", default="8Gi")
    p.add_argument("--mem-limit", default="16Gi")
    p.add_argument("--eph-request", default="4Gi")
    p.add_argument("--eph-limit", default="8Gi")
    args = p.parse_args(argv)
    if args.gpus not in (1, 2, 4, 8):
        p.error("--gpus must be 1, 2, 4, or 8 (NRP: <=2/pod, <=8/Job)")
    return args


def main(argv=None):
    args = parse_args(argv)

    raw = open(args.inventory).read() if args.inventory else sys.stdin.read()
    catalog = json.loads(raw)
    flavors = catalog.get("flavors", catalog if isinstance(catalog, list) else [])

    emit = []
    for flavor in flavors:
        keep, reason = keep_flavor(flavor, args)
        if not keep:
            print(f"[skip] {flavor.get('product','?')} "
                  f"({flavor.get('resource','?')}): {reason}", file=sys.stderr)
            continue
        emit.append(flavor)

    if args.max_jobs and len(emit) > args.max_jobs:
        print(f"[warn] {len(emit)} flavors matched but --max-jobs={args.max_jobs}; "
              f"emitting the first {args.max_jobs}. There is no fair queue on "
              f"Nautilus -- raise --max-jobs deliberately, or narrow with --only.",
              file=sys.stderr)
        emit = emit[:args.max_jobs]
    elif args.max_jobs == 0 and len(emit) > 8:
        print(f"[warn] --max-jobs=0: emitting all {len(emit)} Jobs. A bulk apply "
              f"blocks other users -- make sure this is intended.", file=sys.stderr)

    name_for = job_namer()
    docs = [build_job(fl, name_for(fl.get("product", "gpu")), args)
            for fl in emit]

    for doc in docs:
        sys.stdout.write("---\n")
        sys.stdout.write(json.dumps(doc, indent=2))
        sys.stdout.write("\n")

    print(f"[info] emitted {len(docs)} Job(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
