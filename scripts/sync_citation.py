"""Sync the BibTeX citation block in README.md with pyproject.toml.

Run this after bumping the version or changing the description in
pyproject.toml, before publishing a release:

    python scripts/sync_citation.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"

CITATION_BLOCK = re.compile(
    r"(<!-- citation-start -->\n).*?(\n<!-- citation-end -->)", re.DOTALL
)


def read_pyproject_field(name: str) -> str:
    text = PYPROJECT.read_text()
    match = re.search(rf'^{name}\s*=\s*"(.*)"\s*$', text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find field {name!r} in pyproject.toml")
    return match.group(1)


def main() -> None:
    version = read_pyproject_field("version")
    description = read_pyproject_field("description")

    bibtex = (
        "```bibtex\n"
        "@software{rex2026elyza,\n"
        "  author  = {Rex, Atticus},\n"
        f"  title   = {{elyza: {description.rstrip('.')}}},\n"
        "  year    = {2026},\n"
        "  url     = {https://github.com/atticusdrex/elyza},\n"
        f"  version = {{{version}}}\n"
        "}\n"
        "```"
    )

    readme_text = README.read_text()
    new_text, count = CITATION_BLOCK.subn(rf"\1{bibtex}\2", readme_text)
    if count == 0:
        raise ValueError(
            "Could not find <!-- citation-start --> / <!-- citation-end --> "
            "markers in README.md"
        )
    README.write_text(new_text)
    print(f"Synced citation: version={version!r}, description={description!r}")


if __name__ == "__main__":
    main()
