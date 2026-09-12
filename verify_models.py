#!/usr/bin/env python3
"""
Pre-commit Verification Suite for LemGendary Models Hub.

Enforces four mandatory gates:
1. Python syntax and bytecode compilation (python -m py_compile)
2. Markdown documentation linting (markdownlint-cli)
3. Model card and checkpoint integrity
4. Workspace cleanliness and zero volatile residue
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def log_header(title: str):
    print(f"\n{'=' * 70}")
    print(f" [GATE] {title}")
    print(f"{'=' * 70}")


def get_staged_files() -> list[Path]:
    try:
        res = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if line:
                p = REPO_ROOT / line
                if p.exists():
                    files.append(p)
        return files
    except Exception:
        return []


def check_python_compilation(target_files: list[Path] | None = None) -> bool:
    log_header("GATE 1: Python Syntax & Compilation (python -m py_compile)")
    if target_files:
        py_files = [p for p in target_files if p.suffix.lower() == ".py"]
    else:
        py_files = sorted(list(REPO_ROOT.glob("*.py")))

    if not py_files:
        print("[INFO] No Python files to compile.")
        return True

    print(f"[RUN] Compiling {len(py_files)} Python source files...")
    has_errors = False
    for py_file in py_files:
        cmd = [sys.executable, "-B", "-m", "py_compile", str(py_file)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[FAIL] Syntax error in {py_file.name}:")
            if res.stderr:
                print(res.stderr.strip())
            has_errors = True
        else:
            print(f" [OK] {py_file.name}")

    # Remove ephemeral pycache from compilation
    pycache_dir = REPO_ROOT / "__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir, ignore_errors=True)

    if has_errors:
        print("[FAIL] One or more Python files failed syntax compilation.")
        return False

    print(f"[PASS] All {len(py_files)} Python files compiled cleanly.")
    return True


def check_markdownlint(target_files: list[Path] | None = None) -> bool:
    log_header("GATE 2: Markdown Documentation Linting (markdownlint-cli)")
    config_path = REPO_ROOT / ".markdownlint.json"
    if not config_path.exists():
        print(f"[FAIL] Local .markdownlint.json missing at {config_path}")
        return False

    if target_files:
        md_files = [p for p in target_files if p.suffix.lower() == ".md"]
        if not md_files:
            print("[INFO] No Markdown files to lint.")
            return True
        cmd = ["npx", "markdownlint-cli"] + [str(p.relative_to(REPO_ROOT)) for p in md_files] + [
            "--config",
            str(config_path),
        ]
    else:
        cmd = ["npx", "markdownlint-cli", "**/*.md", "--config", str(config_path)]

    print(f"[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, shell=True)
    if res.returncode != 0:
        print("[FAIL] Markdownlint detected style violations:")
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        return False

    print("[PASS] All Markdown files passed markdownlint with 0 errors.")
    return True


def check_model_integrity() -> bool:
    log_header("GATE 3: Model Directory & Checkpoint Integrity")
    root_readme = REPO_ROOT / "README.md"
    if not root_readme.exists():
        print("[FAIL] Root README.md missing.")
        return False

    model_dirs = sorted([
        d for d in REPO_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in ["__pycache__", ".githooks", "typings"]
    ])
    if not model_dirs:
        print("[FAIL] No model directories found in repository.")
        return False

    print(f"[RUN] Inspecting {len(model_dirs)} model directories...")
    has_errors = False

    for model_dir in model_dirs:
        readme = model_dir / "README.md"
        if not readme.exists():
            print(f"[FAIL] {model_dir.name} missing README.md")
            has_errors = True

        checkpoints_dir = model_dir / "checkpoints"
        if checkpoints_dir.exists():
            pth_files = list(checkpoints_dir.glob("*.pth"))
            for pth in pth_files:
                if pth.stat().st_size == 0:
                    print(f"[FAIL] Zero-byte checkpoint detected: {pth.relative_to(REPO_ROOT)}")
                    has_errors = True

    if has_errors:
        print("[FAIL] Model integrity validation failed.")
        return False

    print(f"[PASS] All {len(model_dirs)} model directories and checkpoints verified.")
    return True


def check_residue(purge: bool = False) -> bool:
    log_header("GATE 4: Workspace Cleanliness & Zero Volatile Residue")
    residue_found = []

    for root, dirs, files in os.walk(REPO_ROOT):
        # Ignore .git directory
        if ".git" in root.split(os.sep):
            continue

        for d in dirs:
            if d in ["__pycache__", ".ipynb_checkpoints"] or d.startswith(".staging_"):
                residue_found.append(Path(root) / d)

        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in [".tmp", ".log", ".kaggle-partial"] or p.name in [".DS_Store", "Thumbs.db", "desktop.ini"]:
                residue_found.append(p)

    if residue_found:
        if purge:
            print(f"[INFO] Purging {len(residue_found)} volatile residue items...")
            for item in residue_found:
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    print(f" [PURGED] {item.relative_to(REPO_ROOT)}")
                except Exception as e:
                    print(f" [ERROR] Could not purge {item.relative_to(REPO_ROOT)}: {e}")
            return True
        else:
            print(f"[FAIL] Found {len(residue_found)} unapproved volatile or temporary files:")
            for item in residue_found:
                print(f"  - {item.relative_to(REPO_ROOT)}")
            print("[HINT] Run with --purge to remove these items automatically.")
            return False

    print("[PASS] Workspace is clean with zero volatile residue.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-commit Verification Suite for LemGendary Models Hub")
    parser.add_argument("--staged", action="store_true", help="Lint only git staged files")
    parser.add_argument("--purge", action="store_true", help="Auto-purge volatile residue")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" LemGendary Models Hub Verification Suite")
    print("=" * 70)

    target_files = get_staged_files() if args.staged else None

    gates = [
        ("Python Compilation", lambda: check_python_compilation(target_files)),
        ("Markdown Documentation", lambda: check_markdownlint(target_files)),
        ("Model Directory & Checkpoint Integrity", check_model_integrity),
        ("Cleanliness & Zero Residue", lambda: check_residue(purge=args.purge)),
    ]

    for gate_name, gate_fn in gates:
        success = gate_fn()
        if not success:
            print(f"\n[ABORT] {gate_name} failed. Fix errors and rerun.\n")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(" [ALL GATES PASSED] LemGendary Models Hub is 100% compliant.")
    print("=" * 70 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
