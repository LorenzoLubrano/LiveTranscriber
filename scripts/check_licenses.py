"""Audit installed dependency licenses (spec §27).

Prints every installed distribution with its license and flags anything that
would conflict with shipping LiveTranscriber as MIT-licensed open source.

Usage::

    python scripts/check_licenses.py
    python scripts/check_licenses.py --all      # include transitive packages
"""

from __future__ import annotations

import argparse
from importlib.metadata import Distribution, distributions

#: Licenses that would force the whole application to become copyleft if the
#: library were linked into it.
BLOCKING = ("gpl-3", "gpl-2", "agpl", "gnu general public")

#: Copyleft licenses that are fine to use but impose packaging conditions.
CONDITIONAL = ("lgpl", "mpl")

#: Proprietary but redistributable: NVIDIA's CUDA runtime libraries ship under
#: their redistribution terms, not an open-source license. They are an optional
#: extra (``pip install -e ".[gpu]"``) and never a required dependency.
PROPRIETARY = ("licenseref-nvidia", "proprietary", "nvidia software license")

#: Build-time-only tools: not part of the distributed application.
BUILD_ONLY = {"pyinstaller", "pyinstaller-hooks-contrib", "pytest", "pytest-qt", "ruff",
              "altgraph", "pefile", "pywin32-ctypes", "iniconfig", "pluggy", "packaging",
              "setuptools", "wheel", "pip"}


def license_of(dist: Distribution) -> str:
    meta = dist.metadata
    for key in ("License-Expression", "License"):
        value = meta.get(key)
        if value and len(value) < 80:
            return value.strip()

    classifiers = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    if classifiers:
        return classifiers[0].split("::")[-1].strip()
    return "UNKNOWN"


def classify(name: str, license_text: str) -> tuple[str, str]:
    """Return (status, note)."""
    lowered = license_text.lower()
    if name.lower() in BUILD_ONLY:
        return "BUILD", "build-time only, not distributed"
    if any(tag in lowered for tag in PROPRIETARY):
        return "PROP", "proprietary but redistributable; optional GPU extra, never bundled by default"
    if any(tag in lowered for tag in CONDITIONAL):
        return "COND", "copyleft: ship license text, keep libraries replaceable"
    if any(tag in lowered for tag in BLOCKING):
        return "BLOCK", "incompatible with MIT distribution"
    if lowered == "unknown":
        return "CHECK", "license metadata missing - verify manually"
    return "OK", ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Dependency license audit")
    parser.add_argument("--all", action="store_true", help="include every installed package")
    args = parser.parse_args()

    rows = []
    for dist in distributions():
        try:
            name = dist.metadata["Name"]
        except KeyError:
            continue
        if not name:
            continue
        license_text = license_of(dist)
        status, note = classify(name, license_text)
        if not args.all and status == "BUILD":
            continue
        rows.append((status, name, dist.version, license_text, note))

    order = {"BLOCK": 0, "CHECK": 1, "PROP": 2, "COND": 3, "BUILD": 4, "OK": 5}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1].lower()))

    print(f"{'STATUS':7s} {'PACKAGE':26s} {'VERSION':12s} LICENSE")
    print("-" * 100)
    for status, name, version, license_text, note in rows:
        print(f"{status:7s} {name:26s} {version:12s} {license_text}")
        if note:
            print(f"{'':7s} -> {note}")

    blocking = [r for r in rows if r[0] == "BLOCK"]
    check = [r for r in rows if r[0] == "CHECK"]

    print("-" * 100)
    print(f"{len(rows)} packages   "
          f"blocking: {len(blocking)}   needs review: {len(check)}   "
          f"conditional: {sum(1 for r in rows if r[0] == 'COND')}   "
          f"proprietary: {sum(1 for r in rows if r[0] == 'PROP')}")

    if blocking:
        print("\nBLOCKING licenses found - these cannot ship inside an MIT application:")
        for _, name, _, license_text, _ in blocking:
            print(f"  - {name}: {license_text}")
        return 1
    if check:
        print("\nPackages needing manual review:")
        for _, name, _, _, _ in check:
            print(f"  - {name}")
    print("\nNo blocking licenses. See THIRD_PARTY_LICENSES.md for obligations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
