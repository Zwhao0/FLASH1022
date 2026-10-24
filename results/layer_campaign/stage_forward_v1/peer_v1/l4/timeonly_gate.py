#!/usr/bin/env python3
"""L4 time-only QC wrapper using the already reviewed v2 gate calculations."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
UPSTREAM = ROOT / "results/layer_campaign/continuation_v2/gates/timeonly_gate.py"
spec = importlib.util.spec_from_file_location("continuation_v2_timeonly_gate_for_l4", UPSTREAM)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load v2 time-only gate: {UPSTREAM}")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def main():
    out = HERE
    ready_path = out / "timeonly_ready.json"
    ready = gate.read_json(ready_path)
    if ready.get("fixed_denominator") != 32 or ready.get("stage") != "L4":
        raise ValueError("L4 time-only bundle must retain the fixed 32-case denominator")
    for name, digest in ready["artifacts"].items():
        path = HERE.parent / "source_freeze.json" if name == "source_freeze.json" else out / name
        if gate.sha(path) != digest:
            raise RuntimeError(f"time-only source artifact changed: {name}")
    layouts = {layout: gate.audit_layout("l2", layout, out)
               for layout in ("legacy", "node_local_pp")}
    stable = all(result["healthy_gate"]["passed"] for result in layouts.values())
    result = {
        "protocol": "peer_v1_l4_timeonly_gate_v1",
        "fixed_denominator": 32, "source_freeze_sha256": ready["source_freeze_sha256"],
        "design_sha256": ready["design_sha256"], "stable_for_formal": stable,
        "weak_cases_are_report_only": True,
        "telemetry_note": "5-second host resource telemetry is auxiliary QC only, never an inference feature",
        "layouts": layouts,
    }
    path = out / "timeonly_gate.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)
    print(json.dumps({"output": str(path), "stable_for_formal": stable,
                      "physical_injection_valid": {name: value["physical_injection_valid_cases"]
                                                    for name, value in layouts.items()}}, indent=2))


if __name__ == "__main__":
    main()
