"""Record environment and aggregate dataset provenance without training or URL access."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from dataset import load_dataset

ROOT = Path(__file__).resolve().parent
REFERENCE = dict(original_rows=235795, deduplicated_rows=235370,
                 phishing=100520, legitimate=134850)


def command(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_details():
    details = {}
    try:
        import torch
        details.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                       cuda_available=torch.cuda.is_available(),
                       mps_available=torch.backends.mps.is_available())
        details["cuda_devices"] = [dict(name=torch.cuda.get_device_name(i),
            capability=list(torch.cuda.get_device_capability(i)))
            for i in range(torch.cuda.device_count())]
    except Exception as exc:
        details["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        details["nvidia_smi"] = command(["nvidia-smi"])
    except (OSError, RuntimeError) as exc:
        details["nvidia_smi_unavailable"] = str(exc)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="local original CSV path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Capture before creating artifacts, including untracked files but never their contents.
    git = dict(commit=command(["git", "rev-parse", "HEAD"], ROOT).strip(),
               status=command(["git", "status", "--porcelain=v1", "--untracked-files=all"], ROOT))
    checksum = sha256(args.data)
    snapshot = command([sys.executable, "-m", "pip", "freeze", "--all"])
    _, counts = load_dataset(args.data)
    if sha256(args.data) != checksum:
        raise RuntimeError("Input changed while reading; no provenance saved")
    actual = dict(original_rows=counts["original"]["rows"],
                  deduplicated_rows=counts["deduplicated"]["rows"],
                  phishing=counts["deduplicated"]["phishing"],
                  legitimate=counts["deduplicated"]["legitimate"])
    versions = {name: metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn")}
    try:
        versions["torch"] = metadata.version("torch")
    except metadata.PackageNotFoundError:
        versions["torch"] = None
    record = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, input=dict(path=str(args.data.resolve()), sha256=checksum),
        git=git, python=dict(version=sys.version, executable=sys.executable),
        dependencies=versions, dependency_snapshot="requirements-environment.txt",
        os=dict(platform=platform.platform(), machine=platform.machine()), gpu=gpu_details(),
        class_mapping={"phishing_label": 0, "training_positive": "phishing",
                       "legitimate": "all other non-missing labels (existing loader mapping)"},
        cleaning="drop missing URL/label; strip URL whitespace; keep first exact URL duplicate; no normalization or subsampling",
        counts=counts, reference=REFERENCE,
        difference_from_reference={k: actual[k] - v for k, v in REFERENCE.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an earlier provenance capture.
    with (args.out / "requirements-environment.txt").open("x") as stream:
        stream.write(snapshot)
    with (args.out / "provenance.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(counts=counts, difference_from_reference=record["difference_from_reference"]), indent=2))


if __name__ == "__main__":
    main()
