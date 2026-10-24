"""Explicit sparse3 + fixed-model startup-GC online launcher."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True,
                        help='Complete approved online plan JSON (including core/peer model bindings).')
    parser.add_argument('--timeout', type=int, default=900,
                        help='Existing data16 run timeout in seconds (default: 900).')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    plan = json.loads(args.plan.read_text())
    if not isinstance(plan, dict):
        parser.error('--plan must contain a JSON object')
    from flash_v1.scripts.data16 import run
    return run(plan, timeout=args.timeout, startup_gc_fixed_model=True)


if __name__ == '__main__':
    main()
