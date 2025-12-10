# File: tools/check_purity.py
"""Command-line interface for purity checking.

Usage:
    poetry run check-purity [--fix] [--explain RULE] [--verbose] [FILES...]

Examples:
    poetry run check-purity                           # Check all Tier 2 files
    poetry run check-purity src/spectralmc/gbm.py     # Check specific file
    poetry run check-purity --fix                     # Auto-fix simple violations
    poetry run check-purity --explain PUR003          # Show rule details
    poetry run check-purity --verbose                 # Show file classification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.purity.classifier import FileClassifier, FileTier
from tools.purity.config import get_project_root, load_purity_config
from tools.purity.messages import explain_rule, format_violation
from tools.purity.rules import check_file_purity


def collect_python_files(
    paths: list[Path] | None,
    project_root: Path,
) -> list[Path]:
    """Collect Python files to check.

    Args:
        paths: Specific files/directories to check (or None for all src/)
        project_root: Project root directory

    Returns:
        List of Python file paths
    """
    if not paths:
        # Default: Check all files in src/spectralmc/
        src_dir = project_root / "src" / "spectralmc"
        return list(src_dir.rglob("*.py"))

    # Check specific paths
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.py"))

    return files


def main() -> int:
    """Run purity checker.

    Returns:
        Exit code (0 = no violations, 1 = violations found)
    """
    parser = argparse.ArgumentParser(
        description="Check SpectralMC business logic for purity violations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run check-purity                      # Check all Tier 2 files
  poetry run check-purity src/spectralmc/gbm.py  # Check specific file
  poetry run check-purity --fix                # Auto-fix simple violations (TODO)
  poetry run check-purity --explain PUR003     # Show rule details
  poetry run check-purity --verbose            # Show file classification

Exit codes:
  0 - No violations found
  1 - Violations found
  2 - Configuration or runtime error
        """,
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Files or directories to check (default: src/spectralmc/)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix simple violations (NOT IMPLEMENTED YET)",
    )
    parser.add_argument(
        "--explain",
        metavar="RULE",
        help="Show detailed explanation for a rule (PUR001, PUR002, etc.)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show file classification and detailed violations",
    )

    args = parser.parse_args()

    # Handle --explain
    if args.explain:
        print(explain_rule(args.explain))
        return 0

    # Handle --fix
    if args.fix:
        print("ERROR: --fix is not implemented yet", file=sys.stderr)
        print(
            "Please manually refactor violations using the guidance in:",
            file=sys.stderr,
        )
        print(
            "  documents/engineering/purity_enforcement.md",
            file=sys.stderr,
        )
        return 2

    # Load configuration
    try:
        config = load_purity_config()
        project_root = get_project_root()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        return 2

    # Create file classifier
    classifier = FileClassifier(
        tier1_patterns=config.get("tier1_infrastructure", []),
        tier3_patterns=config.get("tier3_effects", []),
        project_root=project_root,
    )

    # Collect files to check
    files = collect_python_files(
        paths=args.files if args.files else None,
        project_root=project_root,
    )

    if not files:
        print("No Python files found to check")
        return 0

    print(f"🔍 Checking {len(files)} files for purity violations...")
    if args.verbose:
        print()

    # Check each file
    total_violations = 0
    total_whitelisted = 0
    tier2_files = 0

    for filepath in sorted(files):
        # Classify file
        classification = classifier.classify(filepath)

        if classification is None:
            # File exempt (not in src/spectralmc/)
            if args.verbose:
                print(f"⚪ SKIP: {filepath} (not in src/spectralmc/)")
            continue

        if args.verbose:
            tier_emoji = {
                FileTier.TIER1_INFRASTRUCTURE: "🔧",
                FileTier.TIER2_BUSINESS_LOGIC: "✨",
                FileTier.TIER3_EFFECTS: "⚡",
            }
            emoji = tier_emoji.get(classification.tier, "❓")
            print(f"{emoji} {classification.tier.name}: {filepath.relative_to(project_root)}")
            if args.verbose:
                print(f"    Reason: {classification.reason}")

        # Only check Tier 2 files
        if classification.tier != FileTier.TIER2_BUSINESS_LOGIC:
            continue

        tier2_files += 1

        # Get whitelist for this file
        rel_path = str(filepath.relative_to(project_root))
        file_whitelist = config.get("whitelist", {}).get(rel_path, {})

        # Check for violations
        violations = check_file_purity(
            filepath=filepath,
            tier=classification.tier,
            whitelist=file_whitelist,
        )

        # Report violations
        for violation in violations:
            if violation.whitelisted:
                total_whitelisted += 1
                if args.verbose:
                    print(
                        format_violation(
                            rule_code=violation.rule_code,
                            filepath=str(violation.filepath.relative_to(project_root)),
                            line_number=violation.line_number,
                            context=violation.context,
                            verbose=False,
                        )
                    )
                    print(f"  ✅ WHITELISTED: {violation.whitelist_reason}")
            else:
                total_violations += 1
                print(
                    format_violation(
                        rule_code=violation.rule_code,
                        filepath=str(violation.filepath.relative_to(project_root)),
                        line_number=violation.line_number,
                        context=violation.context,
                        verbose=args.verbose,
                    )
                )
                if args.verbose:
                    print()  # Blank line between violations

    # Summary
    print()
    print(f"✅ Checked {tier2_files} Tier 2 business logic files")
    print(f"❌ Found {total_violations} violations")
    if total_whitelisted > 0:
        print(f"⚠️  {total_whitelisted} whitelisted violations")

    if total_violations > 0:
        print()
        print("Run with --verbose for detailed messages and examples:")
        print("  poetry run check-purity --verbose")
        print()
        print("See documentation:")
        print("  documents/engineering/purity_enforcement.md")
        return 1

    print()
    print("🎉 No purity violations found!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
