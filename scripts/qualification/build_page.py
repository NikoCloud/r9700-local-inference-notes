"""Build the interactive qualification chart from the JSON results.

Layout (relative to the repo root):
  scripts/qualification/turbo_qual_template.html   template with /*..._JSON*/null placeholders
  data/qualification/results*.json                 main window (heretic + two Turbo quants)
  data/qualification/results_unleashed_*.json      later window (Unleashed stock proxy) + drift check
  data/qualification/turbo_qualification.html      output

Run from anywhere: python scripts/qualification/build_page.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "scripts" / "qualification" / "turbo_qual_template.html"
DATA = ROOT / "data" / "qualification"
OUT = DATA / "turbo_qualification.html"


def load(name):
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def merged(main_name, extra_name):
    main, extra = load(main_name), load(extra_name)
    if main is None:
        return extra
    if extra and extra.get("models"):
        main.setdefault("models", {}).update(extra["models"])
        main["later_window"] = {k: extra.get(k) for k in ("started", "finished")}
    return main


def js(obj):
    return "null" if obj is None else json.dumps(obj).replace("</", "<\\/")


html = (TEMPLATE.read_text(encoding="utf-8")
        .replace("/*MTP_JSON*/null", js(merged("results.json", "results_unleashed_mtp.json")))
        .replace("/*NOSPEC_JSON*/null", js(merged("results_nospec.json", "results_unleashed_nospec.json")))
        .replace("/*FIXED_JSON*/null", js(merged("results_fixedlen.json", "results_unleashed_fixedlen.json")))
        .replace("/*DRIFT_JSON*/null", js(load("results_unleashed_drift.json"))))
OUT.write_text(html, encoding="utf-8", newline="\n")
print("wrote", OUT, len(html), "bytes")
