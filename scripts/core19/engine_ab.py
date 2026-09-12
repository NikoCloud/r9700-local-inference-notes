#!/usr/bin/env python3
"""Engine A/B qualification: Nathan Vulkan fork vs production build, heretic Q4_K_S, on the freed R9700.
Evicts production + ComfyUI first, restores both in finally (inside main). Bench port 8085 only."""
import os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.expanduser("~/benchmarks/core19"))
import pld_test as P   # reuse call(), gpu0_mib(), stop_production()

MODEL = P.MODEL
BINS = {"nathan_v075": os.path.expanduser("~/projects/llama.cpp-nathan/build-vulkan/bin/llama-server"),
        "prod_434ddbb": os.path.expanduser("~/projects/llama.cpp-master/build-vulkan/bin/llama-server")}
SPECS = {"none": [], "ngram+mtp": ["--spec-type", "ngram-mod,draft-mtp", "--spec-draft-n-max", "2"]}
DEPTHS = [2000, 32000, 128000]
OUT = os.path.expanduser("~/benchmarks/core19/engine_ab"); os.makedirs(OUT, exist_ok=True)
import json
res = {"started": time.strftime("%F %T"), "model": os.path.basename(MODEL), "grid": {}}
save = lambda: open(OUT+"/results.json","w").write(json.dumps(res,indent=2)+"\n")

def launch(binpath, spec):
    subprocess.run(["pkill","-TERM","-f","llama-server .*--port 8085"],capture_output=True)
    for _ in range(30):
        if subprocess.run(["pgrep","-f","llama-server .*--port 8085"],capture_output=True).returncode!=0: break
        time.sleep(1)
    time.sleep(2)
    args=[binpath,"-m",MODEL,"-dev","Vulkan0","-ngl","99","-c","262144","-ctk","q8_0","-ctv","q8_0",
          "-fa","on","--ctx-checkpoints","2","-np","1","-kvu","--jinja","--reasoning-format","deepseek",
          "-cram","12288","--host","127.0.0.1","--port","8085","--log-timestamps","--log-prefix"]+SPECS[spec]
    log=open(f"{OUT}/srv_{os.path.basename(os.path.dirname(os.path.dirname(binpath)))}_{spec}.log","ab")
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for k in range(150):
        try:
            if urllib.request.urlopen(P.BASE+"/health",timeout=4).status==200: return p,k*2,P.gpu0_mib()
        except Exception: pass
        if p.poll() is not None: return None,None,None
        time.sleep(2)
    return None,None,None

def build_prompt(ntok):  # ~0.75 tok/word filler
    s=("The maintenance log records inspections, discharge pressure, vibration amplitude, and corrective "
       "action for each pump station across the facility over the operating year. ")
    return ("Summarize the following log in one sentence.\n\n"+s*max(1,int(ntok/0.75/len(s.split()))))

def main():
    print("evicting production + comfy...",flush=True)
    subprocess.run(["tmux","kill-session","-t","comfy"],capture_output=True)
    subprocess.run(["pkill","-f","[m]ain.py --port 8188"],capture_output=True)
    if not P.stop_production(): print("ABORT: GPU not freed"); return
    print("GPU0",P.gpu0_mib(),"MiB",flush=True)
    for bname,binpath in BINS.items():
        for spec in SPECS:
            key=f"{bname}/{spec}"; p,ls,mib=launch(binpath,spec)
            if not p: print(key,"SERVER FAIL"); res["grid"][key]={"error":"start"}; save(); continue
            res["grid"][key]={"load_s":ls,"gpu0_mib":mib,"depths":{}}
            P.call(build_prompt(200),gen=32)
            for d in DEPTHS:
                r=P.call(build_prompt(d),gen=512)
                res["grid"][key]["depths"][d]={"pt":r["completion_tokens"] and r.get("prefill_tok_s"),
                    "prefill_tok_s":r.get("prefill_tok_s"),"decode_tok_s":r.get("decode_tok_s"),
                    "ttft_s":r.get("ttft_s"),"accept":r.get("accept"),"ct":r.get("completion_tokens")}
                print(f"  {key:<22} d~{d:>6}: prefill {r.get('prefill_tok_s')} tg {r.get('decode_tok_s')} accept {r.get('accept')} ttft {r.get('ttft_s')}",flush=True)
                save()
            subprocess.run(["pkill","-TERM","-f","llama-server .*--port 8085"],capture_output=True)
    res["finished"]=time.strftime("%F %T"); save()
    print("\n=== decode by depth (tg tok/s) ===")
    for key,v in res["grid"].items():
        if "depths" in v: print(f"  {key:<22} "+" ".join(f"d{d}={v['depths'][d]['decode_tok_s']}" for d in DEPTHS))

if __name__=="__main__":
    try: main()
    except Exception as e: print("CRASH",repr(e))
    finally:
        print("\nrestoring production + comfy...",flush=True)
        subprocess.run(["pkill","-TERM","-f","llama-server .*--port 8085"],capture_output=True); time.sleep(3)
        subprocess.run(["systemctl","--user","start","qwen38.service"])
        for _ in range(60):
            try:
                if urllib.request.urlopen("http://127.0.0.1:8080/health",timeout=4).status==200: break
            except Exception: pass
            time.sleep(5)
        subprocess.run(["tmux","new-session","-d","-s","comfy",
            "cd ~/projects/launcher && python3 launcher.py --start --no-tts --no-whisper --no-speech-gateway --no-sillytavern 2>&1 | tee ~/comfy_launch.log"])
        print("prod 8080:", subprocess.run(["bash","-c","curl -s -o /dev/null -w '%{http_code}' -m4 http://127.0.0.1:8080/health"],capture_output=True,text=True).stdout,"| GPU0",P.gpu0_mib(),"MiB",flush=True)
