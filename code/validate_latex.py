"""
Lightweight static validator for main.tex — runs before arXiv upload.

Checks:
  1. Every \ref{X} has a matching \label{X}
  2. Every \cite{X} (and \cite{X, Y, Z}) has a matching \bibitem{X}
  3. Every \includegraphics{file} refers to an existing file
  4. Brace { } balance
  5. \begin{env} / \end{env} balance for common environments
  6. No leftover placeholder strings
"""

from pathlib import Path
import re
import sys

TEX = Path(
    r"G:\내 드라이브\Hobsidian\01_Projects\07_Sentient_AI\04_출판\논문 초안\latex\main.tex"
)
TEX_DIR = TEX.parent

errors: list[str] = []
warnings: list[str] = []


def check(condition: bool, msg: str, level: str = "ERROR") -> None:
    if not condition:
        if level == "ERROR":
            errors.append(msg)
        else:
            warnings.append(msg)


content = TEX.read_text(encoding="utf-8")

# Strip LaTeX line comments to avoid spurious matches in commented-out code
def strip_comments(text: str) -> str:
    # remove % comments (but keep \%)
    cleaned_lines = []
    for line in text.split("\n"):
        # Find unescaped % and cut there
        idx = 0
        while idx < len(line):
            pct = line.find("%", idx)
            if pct == -1:
                cleaned_lines.append(line)
                break
            if pct > 0 and line[pct - 1] == "\\":
                idx = pct + 1
                continue
            cleaned_lines.append(line[:pct])
            break
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


live = strip_comments(content)

# ---- 1. labels / refs -----------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]+)\}", live))
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", live))

missing_labels = refs - labels
unused_labels = labels - refs

check(
    not missing_labels,
    f"References to undefined labels: {sorted(missing_labels)}",
)
if unused_labels:
    # unused labels are not errors but flag them
    warnings.append(f"Unused labels (informational): {sorted(unused_labels)}")

# ---- 2. citations / bibitems ---------------------------------------------
bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", live))

# \cite may contain a comma-separated list: \cite{a, b, c}
cite_blocks = re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", live)
cited_keys: set[str] = set()
for block in cite_blocks:
    for key in block.split(","):
        cited_keys.add(key.strip())

missing_citations = cited_keys - bibitems
unused_bibitems = bibitems - cited_keys

check(
    not missing_citations,
    f"Citations to undefined bibitems: {sorted(missing_citations)}",
)
if unused_bibitems:
    warnings.append(f"Unused bibitems (informational): {sorted(unused_bibitems)}")

# ---- 3. \includegraphics{file} exists ------------------------------------
graphics_files = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", live)
for gf in graphics_files:
    # Resolve relative to .tex file location
    # Try as-is, then with common extensions if no extension
    candidates = [gf]
    if not any(gf.endswith(ext) for ext in [".pdf", ".png", ".jpg", ".jpeg", ".eps"]):
        candidates += [gf + ".pdf", gf + ".png", gf + ".jpg"]

    found = False
    for cand in candidates:
        cand_path = (TEX_DIR / cand).resolve()
        if cand_path.exists():
            found = True
            break
    check(
        found,
        f"\\includegraphics refers to missing file: {gf} "
        f"(searched relative to {TEX_DIR})",
    )

# ---- 4. brace balance -----------------------------------------------------
# Count unescaped { vs }
open_count = len(re.findall(r"(?<!\\)\{", live))
close_count = len(re.findall(r"(?<!\\)\}", live))
check(
    open_count == close_count,
    f"Brace mismatch: {{ count = {open_count}, }} count = {close_count}",
)

# ---- 5. environment balance ----------------------------------------------
begins = re.findall(r"\\begin\{([a-zA-Z*]+)\}", live)
ends = re.findall(r"\\end\{([a-zA-Z*]+)\}", live)

from collections import Counter

begin_counts = Counter(begins)
end_counts = Counter(ends)
all_envs = set(begin_counts.keys()) | set(end_counts.keys())
for env in sorted(all_envs):
    check(
        begin_counts[env] == end_counts[env],
        f"Environment mismatch for '{env}': "
        f"\\begin = {begin_counts[env]}, \\end = {end_counts[env]}",
    )

# ---- 6. placeholder strings ----------------------------------------------
placeholders = ["[pending]", "TODO", "FIXME", "XXX", "\\fbox"]
for ph in placeholders:
    occurrences = live.count(ph)
    if occurrences > 0:
        # [pending] in ORCID is expected until user fills it
        level = "WARN" if ph == "[pending]" else "ERROR"
        msg = f"Placeholder '{ph}' still present ({occurrences} occurrences)"
        if level == "ERROR":
            errors.append(msg)
        else:
            warnings.append(msg + " — must be replaced before arXiv upload")

# ---- summary --------------------------------------------------------------
print("=" * 65)
print(f"  main.tex static validation")
print(f"  file: {TEX.name}  ({len(content):,} bytes)")
print("=" * 65)
print()
print(f"  Labels   : {len(labels):3d}")
print(f"  Refs     : {len(refs):3d}")
print(f"  Bibitems : {len(bibitems):3d}")
print(f"  Cites    : {len(cited_keys):3d}")
print(f"  Figures  : {len(graphics_files):3d}")
print(f"  Envs     : {len(all_envs):3d}")
print()

if errors:
    print(f"  ERRORS ({len(errors)}):")
    for e in errors:
        print(f"    - {e}")
    print()

if warnings:
    print(f"  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"    - {w}")
    print()

if not errors:
    print("  RESULT: PASS — arXiv-ready (no blocking errors)")
    sys.exit(0)
else:
    print(f"  RESULT: FAIL — {len(errors)} blocking error(s)")
    sys.exit(1)
