"""Download and convert an Open Model Zoo model to OpenVINO IR.

Windows example:
  python scripts\download_model.py --name person-vehicle-bike-detection-2002

Requires:
  pip install openvino-dev

This uses the OMZ tools:
  omz_downloader, omz_converter
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="person-vehicle-bike-detection-2002")
    ap.add_argument("--download-dir", default="omz_download")
    ap.add_argument("--output-dir", default="omz_ir")
    args = ap.parse_args()

    if not shutil.which("omz_downloader") or not shutil.which("omz_converter"):
        print(
            "omz_downloader/omz_converter not found.\n"
            "Install: pip install openvino-dev\n"
            "Or run them from the Scripts folder of your venv.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run(["omz_downloader", "--name", args.name, "--output_dir", str(download_dir)])
    run(["omz_converter", "--name", args.name, "--download_dir", str(download_dir), "--output_dir", str(output_dir)])

    # best-effort: locate IR
    hits = list(output_dir.rglob("*.xml"))
    if hits:
        print("IR XML candidates:")
        for h in hits[:20]:
            print(" -", h)


if __name__ == "__main__":
    main()
