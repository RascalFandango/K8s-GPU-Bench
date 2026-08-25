#!/usr/bin/env python3
"""GPU node inventory scraper for the NRP "Nautilus" Kubernetes cluster.

Reads Kubernetes Node objects (as produced by ``kubectl get nodes -o json``) and
emits a *flavor catalog*: for every distinct ``(GPU product, resource name)``
pair on the cluster, how many nodes carry it, how many are schedulable right now,
total and schedulable GPU counts, VRAM, compute capability, architecture, the
tolerations a Job may safely carry, and whether the flavor sits behind a taint
that needs admin authorization.

Reading node labels is explicitly sanctioned by NRP -- the docs tell users to run
``kubectl get nodes -L nvidia.com/gpu.product`` to choose hardware. See POLICY.md.

INPUT (choose one; precedence is --in-cluster > --file > stdin):
  * stdin (default)  ``kubectl get nodes -o json | gpu_inventory.py``
      Stdin-first is deliberate. It runs from a laptop using the *operator's own*
      credentials, which NRP has bound cluster-wide read on nodes -- so it Just
      Works. A namespace ServiceAccount almost certainly CANNOT list nodes and
      you cannot grant it that yourself (see --in-cluster). Stdin sidesteps the
      whole RBAC problem.
  * --file PATH       read a saved JSON dump, or the bundled test fixture.
  * --in-cluster      talk to the API server with the pod's ServiceAccount token.
                      This is the path that usually returns 403 on nodes.

OUTPUT: --format {md,csv,json}, optional -o FILE (default stdout).

    ============================================================
    ==  allocatable is NOT free.                              ==
    ============================================================
This tool reports what GPUs *exist* and whether their nodes are schedulable in
the coarse sense (Ready and not cordoned). It does NOT -- and from here CANNOT --
report what is currently *free*. Computing free capacity requires the current pod
allocation across the entire cluster, and the operator can only list pods in
their own namespace. So this answers "what hardware is here and how do I request
it," never "what is idle right now." NRP's own Grafana answers the latter. Do not
mistake this output for a scheduler.

    ============================================================
    ==  Tolerations are an allowlist, not a free-for-all.     ==
    ============================================================
NRP policy (Special Use) authorizes users to tolerate only certain taints. We
therefore auto-propose tolerations ONLY for the `nautilus.io/hardware` tier
(arm64, large-gpu, ...). Reservation / system / issue / science-dmz taints are
reported as `restricted` and NEVER auto-tolerated -- "tolerating the wrong taints
... [leads] to policy violations." A restricted flavor is reachable only if an
admin has authorized you for that taint.
"""

import argparse
import csv
import datetime
import io
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

# --- GPU Feature Discovery labels we read (see BUILD-SPEC §6) ----------------
L_PRODUCT = "nvidia.com/gpu.product"
L_MEMORY = "nvidia.com/gpu.memory"          # MiB, as a string
L_CC_MAJOR = "nvidia.com/gpu.compute.major"
L_CC_MINOR = "nvidia.com/gpu.compute.minor"
L_DRV_MAJOR = "nvidia.com/cuda.driver.major"
L_DRV_MINOR = "nvidia.com/cuda.driver.minor"
L_ARCH = "kubernetes.io/arch"               # catches arm64 Grace/GH200 nodes
L_MIG = "nvidia.com/mig.capable"
L_REGION = "topology.kubernetes.io/region"

# The one resource name that is NOT a special request. Everything else under the
# nvidia.com/ prefix is a named flavor you request directly (§2.6).
GENERIC_GPU = "nvidia.com/gpu"

# --- Taint authorization tiers (NRP "Special Use" policy; POLICY.md) ----------
# node.kubernetes.io/* taints are automatic churn (not-ready, unreachable,
# *-pressure) and are ignored. Of the nautilus.io/* taints, only the `hardware`
# tier is broadly tolerable by any user. All other tiers are RESTRICTED: we
# surface them but never auto-generate a toleration, because tolerating a taint
# you are not authorized for is itself a policy violation.
CHURN_TAINT_PREFIX = "node.kubernetes.io/"
TOLERABLE_TAINT_KEYS = ("nautilus.io/hardware",)   # arm64, large-gpu, ...


def classify_taint(taint):
    """Bucket a taint into: churn | tolerable | <restricted-reason>.

    Anything that is not churn and not explicitly tolerable is treated as
    restricted (conservative), tagged with why, so the operator can decide
    whether they hold the authorization to reach that node.
    """
    key = str(taint.get("key", ""))
    if key.startswith(CHURN_TAINT_PREFIX):
        return "churn"
    if key in TOLERABLE_TAINT_KEYS:
        return "tolerable"
    if key.startswith("nautilus.io/reservation"):
        return "reservation"     # only if you are in the approved group
    if key.startswith("nautilus.io/system"):
        return "system"          # never tolerate -- infrastructure nodes
    if key.startswith("nautilus.io/issue"):
        return "issue"           # never tolerate -- nodes with known problems
    if key.startswith("nautilus.io/science-dmz"):
        return "science-dmz"     # only if you need no public Internet
    return "unknown"             # unrecognized -> be conservative, restrict


# --- small helpers -----------------------------------------------------------
def _labels(node):
    return (node.get("metadata") or {}).get("labels") or {}


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _join_ver(major, minor):
    """'8' + '6' -> '8.6'; missing minor keeps just the major; nothing -> ''."""
    if major and minor:
        return f"{major}.{minor}"
    return str(major) if major else ""


def gpu_resources(node):
    """Every nvidia.com/* allocatable resource on the node with count > 0.

    This is the load-bearing bit of §2.6: we do not look up a hardcoded model
    list, we read the allocatable map. ``nvidia.com/gpu`` is the generic pool;
    any other ``nvidia.com/<name>`` (rtx8000, a100, h100, mig-small, ...) is a
    special-request flavor that new hardware joins automatically.
    """
    alloc = (node.get("status") or {}).get("allocatable") or {}
    out = {}
    for key, val in alloc.items():
        if not key.startswith("nvidia.com/"):
            continue
        count = _int(val)
        if count and count > 0:
            out[key] = count
    return out


def is_ready(node):
    for cond in (node.get("status") or {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


def is_cordoned(node):
    return bool((node.get("spec") or {}).get("unschedulable", False))


def node_taints(node):
    return (node.get("spec") or {}).get("taints") or []


def taint_to_toleration(taint):
    """Turn a node taint into the toleration a pod needs to tolerate it."""
    tol = {"key": taint.get("key"), "effect": taint.get("effect")}
    if taint.get("value"):
        tol["operator"] = "Equal"
        tol["value"] = taint["value"]
    else:
        tol["operator"] = "Exists"
    return tol


def node_facts(node):
    """Flatten one Node object into the fields we care about."""
    lb = _labels(node)
    return {
        "name": (node.get("metadata") or {}).get("name", ""),
        "product": lb.get(L_PRODUCT, ""),
        "vram_mib": _int(lb.get(L_MEMORY)),
        "compute_capability": _join_ver(lb.get(L_CC_MAJOR), lb.get(L_CC_MINOR)),
        "driver": _join_ver(lb.get(L_DRV_MAJOR), lb.get(L_DRV_MINOR)),
        "arch": lb.get(L_ARCH, ""),
        "mig_capable": lb.get(L_MIG, ""),
        "region": lb.get(L_REGION, ""),
        "ready": is_ready(node),
        "cordoned": is_cordoned(node),
        "taints": node_taints(node),
        "resources": gpu_resources(node),
    }


# --- aggregation -------------------------------------------------------------
_FILLABLE = ("vram_mib", "compute_capability", "driver", "arch", "mig_capable",
             "region")


def _new_flavor(product, resource):
    return {
        "product": product,
        "resource": resource,
        "gpus_allocatable": 0,
        "gpus_schedulable": 0,
        "vram_mib": None,
        "compute_capability": "",
        "driver": "",
        "arch": "",
        "mig_capable": "",
        "region": "",
        # working sets, stripped before output
        "_nodes": set(),
        "_sched_nodes": set(),
        "_tols": {},
        "_restricted": {},
    }


def build_catalog(nodes):
    """Aggregate raw Node objects into the flavor catalog dict."""
    facts = [node_facts(n) for n in nodes]
    facts = [f for f in facts if f["resources"]]  # drop CPU-only / non-GPU nodes

    flavors = {}
    for f in facts:
        schedulable = f["ready"] and not f["cordoned"]
        for resource, count in f["resources"].items():
            fl = flavors.setdefault((f["product"], resource),
                                    _new_flavor(f["product"], resource))
            fl["_nodes"].add(f["name"])
            fl["gpus_allocatable"] += count
            if schedulable:
                fl["_sched_nodes"].add(f["name"])
                fl["gpus_schedulable"] += count
            # Prefer the first non-empty value we see for descriptive fields;
            # some nodes (e.g. a cordoned one) omit compute/driver labels.
            for key in _FILLABLE:
                if not fl[key] and f[key]:
                    fl[key] = f[key]
            # Classify taints: auto-tolerate only the `hardware` tier; record
            # everything else as restricted (needs admin authorization).
            for taint in f["taints"]:
                tier = classify_taint(taint)
                if tier == "churn":
                    continue
                if tier == "tolerable":
                    tol = taint_to_toleration(taint)
                    fl["_tols"][(tol.get("key"), tol.get("value"),
                                 tol.get("effect"))] = tol
                else:
                    entry = {"key": taint.get("key"), "value": taint.get("value"),
                             "effect": taint.get("effect"), "reason": tier}
                    fl["_restricted"][(entry["key"], entry["value"],
                                       entry["effect"])] = entry

    catalog = [_finalize(fl) for fl in flavors.values()]
    catalog.sort(key=lambda fl: (fl["product"], fl["resource"]))
    return {"generated": _iso_now(), "flavors": catalog}


def _finalize(fl):
    """Compute derived fields and emit keys in the BUILD-SPEC §4.1 order."""
    mib = fl["vram_mib"]
    return {
        "product": fl["product"],
        "resource": fl["resource"],
        # Generic nvidia.com/gpu needs a nodeSelector to pin the product;
        # a named resource targets the hardware on its own (§2.6).
        "needs_node_selector": fl["resource"] == GENERIC_GPU,
        "nodes": len(fl["_nodes"]),
        "schedulable_nodes": len(fl["_sched_nodes"]),
        "gpus_allocatable": fl["gpus_allocatable"],
        "gpus_schedulable": fl["gpus_schedulable"],
        "vram_gib": round(mib / 1024, 1) if mib else None,
        "compute_capability": fl["compute_capability"],
        "arch": fl["arch"],
        # --- fields beyond the documented shape, additive and safe ---
        "driver": fl["driver"],
        "mig_capable": fl["mig_capable"],
        "region": fl["region"],
        # Only `hardware`-tier tolerations end up here -- safe to emit in a Job.
        "tolerations": list(fl["_tols"].values()),
        # True when some node of this flavor sits behind a taint you may not
        # tolerate without admin authorization. Do not auto-schedule onto these.
        "restricted": bool(fl["_restricted"]),
        "restricted_taints": list(fl["_restricted"].values()),
        "node_names": sorted(fl["_nodes"]),
    }


def _iso_now():
    return (datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"))


# --- input -------------------------------------------------------------------
def items_of(data):
    """Accept a NodeList, a bare Node, or a raw list of nodes."""
    if isinstance(data, dict):
        if "items" in data:
            return data["items"] or []
        return [data]
    if isinstance(data, list):
        return data
    return [data]


def fetch_in_cluster():
    """Read nodes via the pod's ServiceAccount. Usually 403s -- see docstring."""
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if not host:
        sys.exit("--in-cluster: KUBERNETES_SERVICE_HOST unset; not in a pod?")
    sa = "/var/run/secrets/kubernetes.io/serviceaccount"
    try:
        token = open(f"{sa}/token").read().strip()
    except OSError as exc:
        sys.exit(f"--in-cluster: cannot read ServiceAccount token: {exc}")
    ctx = ssl.create_default_context(cafile=f"{sa}/ca.crt")
    req = urllib.request.Request(
        f"https://{host}:{port}/api/v1/nodes",
        headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            sys.exit("--in-cluster: 403 Forbidden listing nodes. Your "
                     "ServiceAccount cannot read nodes cluster-wide, and you "
                     "cannot grant it that in a namespace-scoped project. Run "
                     "from a laptop instead:\n"
                     "  kubectl get nodes -o json | gpu_inventory.py")
        raise


def load_nodes(args):
    if args.in_cluster:
        data = fetch_in_cluster()
    elif args.file:
        with open(args.file) as fh:
            data = json.load(fh)
    else:
        if sys.stdin.isatty():
            sys.exit("No input on stdin. Try:\n"
                     "  kubectl get nodes -o json | gpu_inventory.py\n"
                     "or pass --file / --in-cluster.")
        data = json.load(sys.stdin)
    return items_of(data)


# --- formatting --------------------------------------------------------------
def _tol_str(tol):
    key = tol.get("key", "")
    effect = tol.get("effect", "")
    if tol.get("value"):
        return f"{key}={tol['value']}:{effect}"
    return f"{key}:{effect}"


def _tols_cell(tols):
    return ", ".join(_tol_str(t) for t in tols) if tols else "—"


def _restricted_cell(restricted):
    # e.g. "reservation (nautilus.io/reservation=cogrob:NoSchedule)"
    return (", ".join(f"{r['reason']} ({_tol_str(r)})" for r in restricted)
            if restricted else "—")


def fmt_md(catalog):
    flavors = catalog["flavors"]
    node_names = set()
    for fl in flavors:
        node_names.update(fl["node_names"])
    lines = [
        "# GPU Flavor Catalog",
        "",
        f"_Generated {catalog['generated']}_",
        "",
        f"**{len(flavors)} flavors across {len(node_names)} GPU nodes.** "
        "Counts are allocatable capacity, not free capacity — see the notes "
        "below.",
        "",
        "| Product | Resource | Selector? | Nodes (sched/total) | "
        "GPUs (sched/alloc) | VRAM GiB | CC | Arch | Driver | Tolerations | "
        "Restricted |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fl in flavors:
        lines.append("| {product} | `{resource}` | {sel} | {sn}/{tn} | "
                      "{sg}/{ag} | {vram} | {cc} | {arch} | {drv} | {tols} | "
                      "{restr} |".format(
                          product=fl["product"] or "—",
                          resource=fl["resource"],
                          sel="yes" if fl["needs_node_selector"] else "no",
                          sn=fl["schedulable_nodes"], tn=fl["nodes"],
                          sg=fl["gpus_schedulable"], ag=fl["gpus_allocatable"],
                          vram=fl["vram_gib"] if fl["vram_gib"] else "—",
                          cc=fl["compute_capability"] or "—",
                          arch=fl["arch"] or "—",
                          drv=fl["driver"] or "—",
                          tols=_tols_cell(fl["tolerations"]),
                          restr=_restricted_cell(fl["restricted_taints"])))
    lines += [
        "",
        "> **allocatable ≠ free.** This lists GPUs that *exist* and whether "
        "their nodes are Ready and uncordoned — not what is idle right now. "
        "Free capacity needs cluster-wide pod allocation, which a "
        "namespace-scoped operator cannot see. Check NRP's Grafana for "
        "utilization.",
        ">",
        "> **Restricted flavors** sit behind a taint you may not tolerate "
        "without admin authorization (reservation / system / issue / "
        "science-dmz). They are shown for completeness; do not schedule onto "
        "them unless an admin has authorized you. Only `nautilus.io/hardware` "
        "taints are auto-tolerated.",
        "",
    ]
    return "\n".join(lines)


def fmt_csv(catalog):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "product", "resource", "needs_node_selector", "nodes",
        "schedulable_nodes", "gpus_allocatable", "gpus_schedulable", "vram_gib",
        "compute_capability", "arch", "driver", "mig_capable", "region",
        "tolerations", "restricted", "restricted_taints", "node_names",
    ])
    for fl in catalog["flavors"]:
        writer.writerow([
            fl["product"], fl["resource"], fl["needs_node_selector"],
            fl["nodes"], fl["schedulable_nodes"], fl["gpus_allocatable"],
            fl["gpus_schedulable"], fl["vram_gib"], fl["compute_capability"],
            fl["arch"], fl["driver"], fl["mig_capable"], fl["region"],
            ";".join(_tol_str(t) for t in fl["tolerations"]),
            fl["restricted"],
            ";".join(f"{r['reason']}:{_tol_str(r)}"
                     for r in fl["restricted_taints"]),
            ";".join(fl["node_names"]),
        ])
    return buf.getvalue()


def fmt_json(catalog):
    return json.dumps(catalog, indent=2)


FORMATTERS = {"md": fmt_md, "csv": fmt_csv, "json": fmt_json}


# --- entrypoint --------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate Kubernetes GPU nodes into a flavor catalog.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--file", help="read node JSON from a file")
    src.add_argument("--in-cluster", action="store_true",
                     help="read nodes via the pod ServiceAccount (often 403s)")
    parser.add_argument("--format", choices=sorted(FORMATTERS), default="md")
    parser.add_argument("-o", "--output", help="write here instead of stdout")
    args = parser.parse_args(argv)

    catalog = build_catalog(load_nodes(args))
    text = FORMATTERS[args.format](catalog)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
