import json, os
B = os.environ.get("BENCH_DIR", ".")
def load(tag):
    p = f"{B}/vllm_conc2_{tag}.json"
    if os.path.exists(p):
        return json.load(open(p))
    return None
def rows(d):
    return {r["n"]: r for r in d.get("rows", [])}

print("=" * 86)
print(f"{'set':>7} {'n':>4} {'MTP agg':>9} {'NS agg':>9} {'delta':>8} {'MTP ps':>8} {'NS ps':>8} {'MTP tail':>9} {'NS tail':>8}")
for name, tm, tn in (("ladder","h9_tdp350_ladder","h9ns_ladder"),
                     ("rungs","h9_tdp350_rungs","h9ns_rungs")):
    dm, dn = load(tm), load(tn)
    if not dm or not dn:
        continue
    rm, rn = rows(dm), rows(dn)
    for n in sorted(set.intersection(set(rm.keys()), set(rn.keys()))):
        xm, xn = rm[n], rn[n]
        d = (xn["aggregate"] - xm["aggregate"]) / xm["aggregate"] * 100
        print(f"{name[:6]:>7} {n:>4} {xm['aggregate']:>9.1f} {xn['aggregate']:>9.1f} {d:>+7.1f}% {xm['per_stream_mean']:>8.2f} {xn['per_stream_mean']:>8.2f} {xm.get('tail_distinct_min',0):>9.2f} {xn.get('tail_distinct_ratio_min', xn.get('tail_distinct_min',0)):>8.2f}")

print("\n--- depth profile ---")
am, an = None, None
for tag, slot in (("h9_tdp350_ab2", "m"), ("h9ns_ab2", "n")):
    p = f"{B}/vllm_ab2_{tag}.json"
    if os.path.exists(p):
        d = json.load(open(p))
        if slot == "m": am = d
        else: an = d
if am and an:
    rm = {(r["depth"], r["task"]): r for r in am.get("rows", [])}
    rn = {(r["depth"], r["task"]): r for r in an.get("rows", [])}
    print(f"{'depth':>7} {'task':>10} {'PP MTP':>8} {'PP NS':>8} {'TG MTP':>7} {'TG NS':>7} {'tail M':>7} {'tail NS':>8}")
    for k in sorted(set.intersection(set(rm.keys()), set(rn.keys()))):
        xm, xn = rm[k], rn[k]
        print(f"{k[0]:>7} {k[1]:>10} {xm['prefill_tok_s']:>8.0f} {xn['prefill_tok_s']:>8.0f} {xm['decode_tok_s']:>7.1f} {xn['decode_tok_s']:>7.1f} {xm.get('tail_distinct_ratio',0):>7.2f} {xn.get('tail_distinct_ratio',0):>8.2f}")
