"""
SAM2 Import Path Fix

This module fixes the Python import path issue where the local 'sam2' folder
shadows the installed 'sam2' package when running from the sam2_finetune
parent directory.

PROBLEM:
    When running from the DISSERTATION parent directory, Python finds the
    sam2 repo folder instead of the installed sam2 package in site-packages.

    This causes the error:
    "You're likely running Python from the parent directory of the sam2 repository...
    This is not supported since the sam2 Python package could be shadowed..."

SOLUTION:
    Import this module BEFORE any sam2 imports. It will:
    1. Remove problematic paths from sys.path
    2. Verify sam2 is properly installed as a package
    3. Provide a fallback installation command

Usage:
    # At the TOP of your script, BEFORE any other imports that use sam2:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

    from sam2_path_fix import fix_sam2_path, verify_sam2_import
    fix_sam2_path()
    verify_sam2_import()  # Optional, raises error if sam2 not installed

    # Now you can safely import sam2
    from sam2.build_sam import build_sam2
"""

import sys
import os
from pathlib import Path
from typing import List, Optional


def get_problematic_paths() -> List[str]:
    """
    Identify paths that could cause sam2 shadowing.

    Returns list of paths that contain a 'sam2' folder that could
    shadow the installed sam2 package.
    """
    problematic = []

    for p in sys.path:
        # Handle empty string (resolves to cwd) and regular paths
        if p == '':
            path = Path.cwd()
        else:
            path = Path(p)

        # Skip if not a valid directory
        if not path.is_dir():
            continue

        # Check if this directory contains a sam2 folder
        sam2_folder = path / 'sam2'
        if sam2_folder.is_dir():
            # Check for signs it's the sam2 source repo (not installed package)
            # The repo has nested sam2/sam2 structure
            is_repo = (
                (sam2_folder / '.git').exists() or
                (sam2_folder / 'sam2').is_dir() or  # Nested sam2/sam2
                (sam2_folder / 'setup.py').exists() or
                (sam2_folder / 'pyproject.toml').exists() or
                (sam2_folder / 'README.md').exists()  # Repos usually have README
            )
            if is_repo:
                problematic.append(p)

    return problematic


def fix_sam2_path(verbose: bool = False) -> List[str]:
    """
    Fix sys.path to avoid sam2 import shadowing.

    This removes paths that contain a sam2 folder that could shadow
    the installed sam2 package.

    Args:
        verbose: If True, print what paths are being removed

    Returns:
        List of paths that were removed
    """
    problematic = get_problematic_paths()

    if verbose and problematic:
        print(f"[sam2_path_fix] Removing {len(problematic)} problematic paths:")
        for p in problematic:
            print(f"  - {p}")

    # Remove problematic paths
    for p in problematic:
        while p in sys.path:
            sys.path.remove(p)

    return problematic


def verify_sam2_import(raise_error: bool = True) -> bool:
    """
    Verify that sam2 can be imported correctly.

    Args:
        raise_error: If True, raise ImportError if sam2 not available

    Returns:
        True if sam2 is importable, False otherwise
    """
    try:
        import sam2
        # Check it's the real package, not a shadowing folder
        if hasattr(sam2, '__file__') and sam2.__file__:
            # Should be in site-packages, not in DISSERTATION folder
            sam2_path = Path(sam2.__file__).parent

            # Check if it looks like an installed package
            if 'site-packages' in str(sam2_path) or \
               (sam2_path / 'build_sam.py').exists():
                return True

        # Try the actual import we need
        from sam2.build_sam import build_sam2
        return True

    except ImportError as e:
        if raise_error:
            raise ImportError(
                f"SAM2 package not found or incorrectly installed.\n"
                f"Error: {e}\n\n"
                f"To fix this, install SAM2 as an editable package:\n"
                f"  cd sam2_finetune/sam2\n"
                f"  pip install -e .\n\n"
                f"Or from the sam2_finetune directory:\n"
                f"  pip install -e sam2/\n"
            )
        return False


def ensure_sam2_available(verbose: bool = False) -> None:
    """
    Convenience function that fixes path and verifies import.

    Call this at the start of your script before any sam2 imports.

    Args:
        verbose: If True, print diagnostic information
    """
    removed = fix_sam2_path(verbose=verbose)

    if verbose and removed:
        print(f"[sam2_path_fix] Removed {len(removed)} problematic paths")

    if not verify_sam2_import(raise_error=False):
        print("\n" + "="*70)
        print("SAM2 INSTALLATION REQUIRED")
        print("="*70)
        print("\nThe sam2 package is not properly installed.")
        print("\nTo fix this, run one of these commands:\n")
        print("  Option 1 (from sam2_finetune directory):")
        print("    pip install -e sam2/")
        print()
        print("  Option 2 (from sam2 directory):")
        print("    cd sam2")
        print("    pip install -e .")
        print()
        print("="*70 + "\n")
        raise ImportError("SAM2 package not installed. See instructions above.")

    if verbose:
        print("[sam2_path_fix] SAM2 import verified successfully")


# Auto-fix on import (silent by default)
# This runs when this module is imported
_removed = fix_sam2_path(verbose=False)
