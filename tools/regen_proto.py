# tools/regen_proto.py
"""Regenerate protobuf code from .proto sources.

Usage: poetry run regen-proto
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Regenerate protobuf code from .proto sources."""
    proto_dir = Path("/spectralmc/src/spectralmc/proto")
    output_dir = Path("/opt/spectralmc_proto")

    if not proto_dir.exists():
        print(f"Error: Proto source directory not found: {proto_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    proto_files = list(proto_dir.glob("*.proto"))

    print(f"Found {len(proto_files)} .proto files:")
    for f in proto_files:
        print(f"  - {f.name}")

    cmd = [
        "protoc",
        f"--python_out={output_dir}",
        f"--proto_path={proto_dir}",
    ] + [str(f) for f in proto_files]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\n✓ Generated protobuf code in {output_dir}/")
    for generated_file in sorted(output_dir.glob("*_pb2.py*")):
        print(f"  - {generated_file.name}")

    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Generated protobuf modules for SpectralMC."""\n')
        print(f"\n✓ Created {init_file}")

    print("\n✓ Protobuf regeneration complete")


if __name__ == "__main__":
    main()
