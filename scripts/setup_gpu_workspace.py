"""Create workspace directories idempotently; optional manifest inventories existing data only."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_agent.paths import WorkspacePaths, project_path
from recovery_agent.training.data import create_delivery, verify_delivery

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-delivery", action="store_true")
    parser.add_argument("--manifest", help="New manifest path; existing manifests are verified, not overwritten")
    args = parser.parse_args()
    env = dict(os.environ)
    if args.data_root:
        env["RECOVERY_DATA_ROOT"] = args.data_root
    workspace = WorkspacePaths.from_env(env)
    print(json.dumps(workspace.public(), indent=2))
    if args.dry_run:
        return 0
    workspace.initialize()
    if args.write_delivery:
        destination = project_path(args.manifest) if args.manifest else workspace.delivery
        if destination.exists():
            verify_delivery(workspace, destination)
            print("Existing delivery manifest verified; no replacement")
        else:
            manifest = create_delivery(workspace, destination)
            print(f"Delivery SHA-256 inventory: {len(manifest['files'])} files -> {destination}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
