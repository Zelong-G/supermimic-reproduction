"""Read-only process GPU-memory monitor; does not launch a GPU workload."""

import argparse
import subprocess
import time
from pathlib import Path

from common import output_path, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = output_path(args.out)
    if out.exists():
        raise FileExistsError(out)
    samples = []
    peak = 0
    while Path(f"/proc/{args.pid}").exists():
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        values = [line.split(",") for line in result.splitlines() if line.strip()]
        used = next((int(v[1].strip()) for v in values if int(v[0]) == args.pid), 0)
        peak = max(peak, used)
        samples.append({"unix_time": time.time(), "process_used_mib": used})
        save_json(
            out,
            {
                "pid": args.pid,
                "status": "sampling",
                "peak_observed_process_mib": peak,
                "scope": "Observed nvidia-smi process allocation; monitor attached during first compile, so this is an observed lower bound. PyTorch max-memory statistics separately cover the whole update.",
                "samples": samples,
            },
        )
        time.sleep(1)
    save_json(
        out,
        {
            "pid": args.pid,
            "status": "finished",
            "peak_observed_process_mib": peak,
            "scope": "Observed nvidia-smi process allocation, attached during compile; see full_check.json for complete-update PyTorch peaks.",
            "samples": samples,
        },
    )


if __name__ == "__main__":
    main()
