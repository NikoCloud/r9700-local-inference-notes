#!/usr/bin/env python3
"""Build data/core19/core19_ab.html: a self-contained chart page for the Core-19 heretic vs Turbo A/B.

Inputs (all in data/core19/): core19_report.json, per_trial_server_stats.json, speed_by_depth.json.
No network access needed to view the output; no external libraries.

usage: python scripts/core19/build_core19_page.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "data" / "core19"

report = json.loads((D / "core19_report.json").read_text(encoding="utf-8"))
server = json.loads((D / "per_trial_server_stats.json").read_text(encoding="utf-8"))
depth = json.loads((D / "speed_by_depth.json").read_text(encoding="utf-8"))

arms = {k: report["arms"][k] for k in ("heretic", "turbo")}
tasks = sorted({t for a in arms.values() for t in a["tasks"]})


def slim(x):
    if not x:
        return None
    if x.get("infra_error"):
        state = "infra"
    elif x.get("reward") == 1.0:
        state = "pass"
    elif x.get("exception") == "AgentTimeoutError":
        state = "timeout"
    else:
        state = "fail"
    return {"state": state, "min": x.get("agent_minutes"), "out": x.get("output_tokens"), "steps": x.get("steps"),
            "smoke": bool(x.get("reused_from_smoke"))}


rows = []
for t in tasks:
    row = {"task": t}
    for arm in arms:
        v = arms[arm]["tasks"].get(t, {})
        row[arm + "_a1"] = slim(v.get("a1"))
        row[arm + "_a2"] = slim(v.get("a2"))
        s = server.get(arm, {}).get("a1:" + t)
        row[arm + "_peak"] = s["peak_context"] if s else None
    rows.append(row)

payload = {"rows": rows, "summary": {k: arms[k]["summary"] for k in arms}, "depth": depth}

HTML = r"""<title>Core-19 heretic vs Turbo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#fbfaf7;--fg:#1d1f23;--mut:#6b6f76;--line:#e3e1dc;--card:#ffffff;
--pass:#2e8b57;--fail:#c0392b;--timeout:#d98a1f;--infra:#9aa0a6;--her:#2f6fb3;--tur:#b5562b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#e8e6e1;--mut:#9aa0a6;--line:#2c3036;--card:#1c1f23;
--pass:#4fb37a;--fail:#e06655;--timeout:#e6a94a;--infra:#6f757c;--her:#6ea3e0;--tur:#e08a5c}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#e8e6e1;--mut:#9aa0a6;--line:#2c3036;--card:#1c1f23;
--pass:#4fb37a;--fail:#e06655;--timeout:#e6a94a;--infra:#6f757c;--her:#6ea3e0;--tur:#e08a5c}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:34px 0 8px}
.sub{color:var(--mut);margin:0 0 18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.tile .k{color:var(--mut);font-size:12px}.tile .v{font-size:20px;font-variant-numeric:tabular-nums}
.tile .v b{color:var(--her)}.tile .v i{color:var(--tur);font-style:normal}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--mut);font-size:12px}
.pill{display:inline-block;min-width:54px;padding:1px 6px;border-radius:6px;color:#fff;font-size:12px;text-align:center}
.pass{background:var(--pass)}.fail{background:var(--fail)}.timeout{background:var(--timeout)}.infra{background:var(--infra)}
.m{color:var(--mut);font-size:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut);font-size:12px;margin:6px 0}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--mut);font-size:11px}
.note{color:var(--mut);font-size:12px;margin-top:6px}
</style>
<main>
<h1>Core-19 agentic A/B: heretic vs Turbo (Qwen3.8-27B, Q4_K_S)</h1>
<p class="sub">Same server line (llama.cpp 434ddbb Vulkan, MTP n_max 2, 262k context, q8_0 KV), same harness (Harbor 0.20.0, Terminus-2), one task at a time. Attempt 1 drives all efficiency numbers.</p>
<div class="tiles" id="tiles"></div>

<h2>Per task</h2>
<div class="legend"><span><span class="sw pass"></span>pass</span><span><span class="sw fail"></span>fail</span><span><span class="sw timeout"></span>3 h timeout</span><span><span class="sw infra"></span>harness setup error (agent never ran)</span></div>
<div class="wrap"><table id="tbl"></table></div>
<p class="note">"Setup error": Terminus-2 installs tmux inside each container with a 120 s per-command limit. Overnight the plain-HTTP Ubuntu apt mirrors were hanging, so every ubuntu:24.04 task image failed setup in the Turbo arm. Heretic ran earlier the same day with no such errors.</p>

<h2>Minutes per task (attempt 1)</h2>
<div class="legend"><span><span class="sw" style="background:var(--her)"></span>heretic</span><span><span class="sw" style="background:var(--tur)"></span>Turbo</span><span>hollow bar = failed / timed out</span></div>
<div class="wrap"><svg id="mins"></svg></div>

<h2>Output tokens per task (attempt 1)</h2>
<div class="wrap"><svg id="toks"></svg></div>

<h2>In-run speed by context depth (both attempts, all requests)</h2>
<div class="wrap"><svg id="depth"></svg></div>
<p class="note">Token-weighted per depth bin from each arm's server log. Decode includes MTP speculation. Turbo's 90k+ bin is almost entirely one timed-out retry (mailman) that filled the context to 262,143 tokens.</p>
</main>
<script>
const P = __PAYLOAD__;
const $ = id => document.getElementById(id);
const S = P.summary;
const tiles = [
  ["pass@1", `<b>${S.heretic.pass1}</b> / 19 · <i>${S.turbo.pass1}</i> / 19`],
  ["pass@1, excluding setup errors", `<b>${S.heretic.pass1_excluding_infra}</b> · <i>${S.turbo.pass1_excluding_infra}</i>`],
  ["pass@2", `<b>${S.heretic.pass2}</b> · <i>${S.turbo.pass2}</i>`],
  ["minutes per solved task", `<b>${S.heretic.minutes_per_solved_a1}</b> · <i>${S.turbo.minutes_per_solved_a1}</i>`],
  ["output tokens per solved task", `<b>${(S.heretic.output_tokens_per_solved_a1/1000).toFixed(1)}k</b> · <i>${(S.turbo.output_tokens_per_solved_a1/1000).toFixed(1)}k</i>`],
  ["median solve (min · tokens)", `<b>${S.heretic.median_minutes_solved_a1} · ${(S.heretic.median_output_tokens_solved_a1/1000).toFixed(1)}k</b><br><i>${S.turbo.median_minutes_solved_a1} · ${(S.turbo.median_output_tokens_solved_a1/1000).toFixed(1)}k</i>`],
];
$("tiles").innerHTML = tiles.map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("")
  + `<div class="tile"><div class="k">colour key</div><div class="v"><b>heretic</b> · <i>Turbo</i></div></div>`;

const cell = c => !c ? `<span class="m">–</span>` :
  `<span class="pill ${c.state}">${c.state}</span> <span class="m">${c.min ?? ""}${c.min != null ? " min" : ""}${c.out ? " · " + (c.out/1000).toFixed(1) + "k" : ""}${c.smoke ? " (smoke)" : ""}</span>`;
$("tbl").innerHTML = `<tr><th>task</th><th>heretic attempt 1</th><th>heretic attempt 2</th><th>Turbo attempt 1</th><th>Turbo attempt 2</th><th>peak context a1 (H · T)</th></tr>` +
  P.rows.map(r => `<tr><td>${r.task}</td><td>${cell(r.heretic_a1)}</td><td>${cell(r.heretic_a2)}</td><td>${cell(r.turbo_a1)}</td><td>${cell(r.turbo_a2)}</td>
  <td class="m">${r.heretic_peak ? (r.heretic_peak/1000).toFixed(1)+"k" : "–"} · ${r.turbo_peak ? (r.turbo_peak/1000).toFixed(1)+"k" : "–"}</td></tr>`).join("");

function bars(svgId, field, unit, fmt) {
  const rows = P.rows, W = 1040, left = 230, rowH = 26, H = rows.length * rowH + 30;
  const vals = rows.flatMap(r => [r.heretic_a1, r.turbo_a1]).filter(c => c && c[field]).map(c => c[field]);
  const max = Math.max(...vals), sc = v => (W - left - 90) * v / max;
  let s = "";
  rows.forEach((r, i) => {
    const y = 14 + i * rowH;
    s += `<text x="${left - 8}" y="${y + 12}" text-anchor="end">${r.task}</text>`;
    [["heretic_a1", "var(--her)", 0], ["turbo_a1", "var(--tur)", 11]].forEach(([k, col, dy]) => {
      const c = r[k];
      if (!c || c.state === "infra" || !c[field]) {
        s += `<text x="${left + 4}" y="${y + dy + 9}" style="font-size:10px">${c && c.state === "infra" ? "setup error" : "–"}</text>`;
        return;
      }
      const w = Math.max(1, sc(c[field])), solid = c.state === "pass";
      s += `<rect x="${left}" y="${y + dy}" width="${w}" height="9" rx="2" fill="${solid ? col : "none"}" stroke="${col}" stroke-width="1.2"/>`;
      s += `<text x="${left + w + 5}" y="${y + dy + 8}" style="font-size:10px">${fmt(c[field])}</text>`;
    });
  });
  const svg = $(svgId); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.innerHTML = s;
}
bars("mins", "min", "min", v => v.toFixed(1));
bars("toks", "out", "tok", v => (v / 1000).toFixed(1) + "k");

(function depthChart() {
  const bins = P.depth.bins, W = 1040, H = 300, L = 60, R = 70, T = 20, B = 40;
  const x = i => L + (W - L - R) * (i + 0.5) / bins.length;
  const yT = v => T + (H - T - B) * (1 - v / 70), yA = v => T + (H - T - B) * (1 - (v - 0.5) / 0.5);
  let s = "";
  for (let v = 0; v <= 70; v += 10) s += `<line x1="${L}" x2="${W - R}" y1="${yT(v)}" y2="${yT(v)}" stroke="var(--line)"/><text x="${L - 8}" y="${yT(v) + 4}" text-anchor="end">${v}</text>`;
  for (let a = 0.5; a <= 1.0001; a += 0.1) s += `<text x="${W - R + 8}" y="${yA(a) + 4}">${a.toFixed(1)}</text>`;
  s += `<text x="${L}" y="${T - 6}">decode tok/s (solid)</text><text x="${W - R}" y="${T - 6}" text-anchor="end">MTP acceptance (dashed)</text>`;
  bins.forEach((b, i) => s += `<text x="${x(i)}" y="${H - B + 18}" text-anchor="middle">${b.label}</text>`);
  [["heretic", "var(--her)"], ["turbo", "var(--tur)"]].forEach(([arm, col]) => {
    const pts = bins.map((b, i) => b[arm] ? [x(i), b[arm].tg, b[arm].acc, b[arm].n] : null).filter(Boolean);
    s += `<polyline fill="none" stroke="${col}" stroke-width="2" points="${pts.map(p => p[0] + "," + yT(p[1])).join(" ")}"/>`;
    s += `<polyline fill="none" stroke="${col}" stroke-width="1.5" stroke-dasharray="5 4" points="${pts.map(p => p[0] + "," + yA(p[2])).join(" ")}"/>`;
    pts.forEach(p => s += `<circle cx="${p[0]}" cy="${yT(p[1])}" r="3.5" fill="${col}"><title>${arm}: ${p[1]} tok/s, acceptance ${p[2]}, ${p[3]} requests</title></circle>`);
  });
  const svg = $("depth"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.innerHTML = s;
})();
</script>
"""

out = D / "core19_ab.html"
out.write_text(HTML.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8", newline="\n")
print("wrote", out)
