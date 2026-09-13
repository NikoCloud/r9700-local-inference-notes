#!/usr/bin/env python3
"""Link + freshness check for these notes.

Checks (stdlib only, read-only):
  1. Relative links in every markdown file resolve to a real file or directory.
  2. FINDINGS.md lists entries newest-first, and the README's
     "last material update" date is not older than the newest ledger entry.
  3. Every docs/NN-*.md file is listed in docs/README.md.

Usage:  python scripts/check_docs.py     (from the repo root; works from anywhere)
Exit:   0 = clean (warnings allowed);   1 = broken links / missing docs / index gaps.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
errors = []
warnings = []


def md_files():
    """Every tracked markdown file, minus .git and agent scratch space."""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".agent")]
        for fn in filenames:
            if fn.lower().endswith(".md"):
                yield os.path.join(dirpath, fn)


def strip_code_fences(text):
    """Drop fenced code blocks so example snippets can't trip the link check."""
    out, in_fence = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def check_links():
    """Relative markdown links (and image embeds) must resolve on disk."""
    n = 0
    for path in md_files():
        rel = os.path.relpath(path, ROOT)
        text = strip_code_fences(open(path, encoding="utf-8", errors="replace").read())
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            tpath = target.split("#", 1)[0]
            if not tpath or any(c in tpath for c in "<>{}"):
                continue  # placeholder inside a template
            n += 1
            full = os.path.normpath(os.path.join(os.path.dirname(path), tpath))
            if not os.path.exists(full):
                errors.append(f"broken link: {rel} -> {target}")
    return n


def check_findings_order():
    """FINDINGS.md exists, has dated entries, and they are newest-first."""
    fp = os.path.join(ROOT, "FINDINGS.md")
    if not os.path.exists(fp):
        errors.append("FINDINGS.md is missing")
        return None
    dates = re.findall(
        r"^## (\d{4}-\d{2}-\d{2})", open(fp, encoding="utf-8").read(), re.M
    )
    if not dates:
        errors.append("FINDINGS.md has no '## YYYY-MM-DD' entry headings")
        return None
    for older, newer in zip(dates, dates[1:]):
        if newer > older:
            warnings.append(
                f"FINDINGS.md: entry {newer} sits below older entry {older} (newest goes on top)"
            )
    return dates[0]


def check_staleness(newest_finding):
    """README 'last material update' must not lag the newest ledger entry."""
    rp = os.path.join(ROOT, "README.md")
    if not os.path.exists(rp):
        errors.append("README.md is missing")
        return
    m = re.search(r"[Ll]ast material update:?\s*\*{0,2}(\d{4}-\d{2}-\d{2})",
                  open(rp, encoding="utf-8").read())
    if not m:
        warnings.append("README.md has no 'Last material update: YYYY-MM-DD' line")
    elif newest_finding and m.group(1) < newest_finding:
        warnings.append(
            f"README.md digest date {m.group(1)} is older than newest FINDINGS entry {newest_finding}"
        )
    sp = os.path.join(ROOT, "STATE.md")
    if os.path.exists(sp):
        sm = re.search(r"As of\s*\*{0,2}(\d{4}-\d{2}-\d{2})",
                       open(sp, encoding="utf-8").read())
        if sm:
            print(f"  STATE.md 'as of': {sm.group(1)}")


def check_docs_index():
    """Every docs/NN-*.md file must be linked from docs/README.md."""
    docs = os.path.join(ROOT, "docs")
    idx = os.path.join(docs, "README.md")
    if not os.path.exists(idx):
        errors.append("docs/README.md (the index) is missing")
        return
    idx_text = open(idx, encoding="utf-8").read()
    for fn in sorted(os.listdir(docs)):
        if fn.endswith(".md") and not fn.startswith("_") and fn != "README.md":
            if fn not in idx_text:
                errors.append(f"docs/{fn} is not listed in docs/README.md")


def main():
    print(f"checking {ROOT} ...")
    n_links = check_links()
    newest = check_findings_order()
    check_staleness(newest)
    check_docs_index()
    print(f"  {n_links} relative links checked")
    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  FAIL: {e}")
    if errors:
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        sys.exit(1)
    print(f"OK: no errors, {len(warnings)} warning(s)")
    sys.exit(0)


if __name__ == "__main__":
    main()
