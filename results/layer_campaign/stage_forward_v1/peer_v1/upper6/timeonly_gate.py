#!/usr/bin/env python3
"""Reuse the v2 30-step stability/time-only audit for upper6 fixed plans."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SOURCE_FREEZE = HERE.parent / "source_freeze.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from results.layer_campaign.continuation_v2.gates.timeonly_gate import audit_layout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main(stage):
    if stage not in ("l1", "l2"):
        raise SystemExit("usage: timeonly_gate.py l1|l2")
    out = HERE / stage
    ready = read(out / "timeonly_ready.json")
    if ready.get("source_freeze_sha256") != sha(SOURCE_FREEZE):
        raise RuntimeError("peer source freeze changed after time-only plan")
    for name, digest in ready["artifacts"].items():
        path = SOURCE_FREEZE if name == "source_freeze.json" else out / name
        if sha(path) != digest:
            raise RuntimeError(f"time-only input changed: {name}")
    layouts = {layout: audit_layout(stage, layout, out) for layout in ("legacy", "node_local_pp")}
    stable = all(item["healthy_gate"]["passed"] for item in layouts.values())
    result = {"protocol": f"upper6_{stage}_timeonly_gate_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_freeze_sha256": ready["source_freeze_sha256"],
              "design_sha256": ready["design_sha256"],
              "fixed_denominator": 2 if stage == "l1" else 4,
              "stable_for_formal": stable,
              "telemetry_not_inference_input": True,
              "layouts": layouts}
    target = out / "timeonly_gate.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(target)
    print(json.dumps({"output": str(target.relative_to(ROOT)), "stable_for_formal": stable,
                      "healthy_gate": {k: v["healthy_gate"]["passed"] for k, v in layouts.items()},
                      "physical_valid": {k: v["physical_injection_valid_cases"] for k, v in layouts.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: timeonly_gate.py l1|l2")
    main(sys.argv[1])
