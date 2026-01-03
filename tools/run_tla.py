"""Run TLC against a TLA+ module using the container-provided tooling."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_TLA_JAR = "/opt/tla/tla2tools.jar"


def _resolve_tla_jar() -> Path:
    jar = Path(os.environ.get("TLA_JAR", _DEFAULT_TLA_JAR))
    if not jar.exists():
        raise FileNotFoundError(
            "TLA+ tools jar not found. Set TLA_JAR or install to /opt/tla/tla2tools.jar"
        )
    return jar


def _resolve_spec_paths(spec: Path, config: Path | None) -> tuple[Path, Path | None, Path]:
    spec_path = spec.resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    spec_dir = spec_path.parent
    module_name = spec_path.stem
    config_path: Path | None
    if config is None:
        config_path = None
    else:
        config_path = config
        if not config_path.is_absolute():
            config_path = (spec_dir / config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    return spec_dir, config_path, Path(module_name)


def _build_command(
    jar: Path, module_name: Path, config: Path | None, workers: str, extra: list[str]
) -> list[str]:
    cmd = [
        "java",
        "-cp",
        str(jar),
        "tlc2.TLC",
        "-workers",
        workers,
    ]
    if config is not None:
        cmd.extend(["-config", str(config)])
    cmd.extend(extra)
    cmd.append(str(module_name))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TLC against a TLA+ module using the container tooling."
    )
    parser.add_argument("--spec", required=True, help="Path to the .tla module file")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a .cfg file (relative to the spec directory)",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="TLC workers count (default: auto)",
    )
    parser.add_argument(
        "--tlc-arg",
        action="append",
        default=[],
        help="Additional TLC argument (repeatable)",
    )
    args = parser.parse_args()

    try:
        jar = _resolve_tla_jar()
        spec_dir, config_path, module_name = _resolve_spec_paths(
            Path(args.spec), Path(args.config) if args.config else None
        )
        cmd = _build_command(jar, module_name, config_path, args.workers, args.tlc_arg)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    result = subprocess.run(cmd, cwd=spec_dir)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
