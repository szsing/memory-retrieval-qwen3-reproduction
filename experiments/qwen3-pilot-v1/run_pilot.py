"""Development-only controlled pilot; no held-out queries. Fixed parameters, no tuning."""
import sys, os, json, time, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"rsi-memory"),str(ROOT/"external-memory")]
import rsi, bridge
OUT=ROOT/"experiments/qwen3-pilot-v1"; OUT.mkdir(parents=True,exist_ok=True)
rsi.lm.CACHE=str(OUT/"model-cache.jsonl")
def dump(name,obj): 
    p=OUT/name; tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)); tmp.replace(p)
def event(stage,**kw):
    row=dict(time=time.strftime("%Y-%m-%d %H:%M:%S"),stage=stage,**kw)
    with (OUT/"progress.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
    print(json.dumps(row),flush=True)
def generate(items, template, field, kind):
    output=[]
    def one(ch):
        prompt=template.format(lines=rsi.numbered([x[:600] for x in ch]),reqs=rsi.numbered(ch))
        j=rsi.jcall(prompt)
        rows=[j] if isinstance(j,dict) and "id" in j else j
        valid=isinstance(rows,list) and all(isinstance(x,dict) for x in rows)
        ids=[x.get("id") for x in rows] if valid else []
        valid=valid and len(ids)==len(ch) and set(map(str,ids))==set(map(str,range(len(ch))))
        if valid:
            vals=rsi.by_id(rows,len(ch),field)
            valid=all(isinstance(v,list) and all(isinstance(x,str) for x in v) for v in vals) if field=="p" else all(isinstance(v,str) for v in vals)
        if not valid:
            event("schema_retry",kind=kind,batch=len(ch))
            if len(ch)==1: raise ValueError("Invalid model schema: "+kind)
            m=len(ch)//2; return one(ch[:m])+one(ch[m:])
        return [v[:8] if field=="p" else v for v in vals]
    for start in range(0,len(items),8):
        output+=one(items[start:start+8])
        event("generation",kind=kind,done=len(output),total=len(items))
        dump(kind+".json",output)
    return output
def rank(score,pool): return list(map(int,pool[np.argsort(-score[pool],kind="stable")]))
def rr(score,pool):
    out=np.zeros(len(score)); out[rank(score,pool)]=1/(60+np.arange(len(pool))); return out
# Regression checks: outsiders cannot change in-pool RRF; rerank fallback independent of join.
p=np.array([0,1]); assert np.array_equal(rr(np.array([.2,.1,9]),p)[:2],rr(np.array([.2,.1,-9]),p)[:2])
def rerank(base,picks,candidates): return list(dict.fromkeys([candidates[i] for i in picks]+base))
assert rerank([0,1,2],[1],[0,2])[:3]==[2,0,1]
b=rsi.bed_inmind(); selected=sorted(b["stream"]+b["gate"],key=lambda t:rsi.h(t["id"]))[:20]
tasks=[dict(id=t["id"],q=t[k],target=t["target"],kind=kind) for kind,k in [("indirect","q"),("direct","naive")] for t in selected]
lines=b["lines"]; qs=[t["q"] for t in tasks]
manifest=dict(protocol="development-pilot-v1",source_commit="d62aa408a4e0b9515eed9313663a025a07b08497",model=rsi.lm.MODEL,n_tasks=len(tasks),selected_ids=[t["id"] for t in selected],selection="first 20 by fixed original task-ID hash, stream+gate only",held_out_access=False,k=6,w=2,rrf_constant=60,prompt_version=1,seed=42,context=8192,max_new_tokens=2048,thinking=False,batch=8,schema_failure="split batch recursively, abort if singleton invalid",protocols=["dense365","official241"],note="official241 ranks and picks only background + own fact; write-time transforms computed once per line",started=time.strftime("%Y-%m-%d %H:%M:%S"))
dump("manifest.json",manifest); dump("tasks.json",tasks)
st=rsi.Store(dict(lines=lines,nb=None)); S=st.scores(qs)
results={}
def save_method(name,sc=None, rankings=None):
    rows=[]
    for protocol in manifest["protocols"]:
        for i,t in enumerate(tasks):
            pool=np.arange(len(lines)) if protocol=="dense365" else np.array(list(range(b["nb"]))+[t["target"]])
            order=rank(sc[i],pool) if rankings is None else rankings[(protocol,i)]
            rows.append(dict(protocol=protocol,kind=t["kind"],id=t["id"],hit=int(t["target"] in order[:6]),top6=order[:6],target=t["target"]))
    results[name]=rows; dump("results.json",results)
    agg={p+"_"+k:[sum(r["hit"] for r in rows if r["protocol"]==p and r["kind"]==k),sum(r["protocol"]==p and r["kind"]==k for r in rows)] for p in manifest["protocols"] for k in ["indirect","direct"]}
    dump("summary.json",{n:{p+"_"+k:[sum(r["hit"] for r in rs if r["protocol"]==p and r["kind"]==k),sum(r["protocol"]==p and r["kind"]==k for r in rs)] for p in manifest["protocols"] for k in ["indirect","direct"]} for n,rs in results.items()})
    event("method_done",method=name,scores=agg,calls=rsi.lm.STATS.copy())
save_method("dense",sc=S)
LP=generate(lines,bridge.PROPL,"p","line_properties")
QP=generate(qs,bridge.PROPQ,"p","query_properties")
B=bridge.PropIndex(LP).bonus(QP)
gate=generate(qs,bridge.GATEQ,"k","gate")
if not all(g.strip().lower() in ("help","recall") for g in gate): raise ValueError("invalid gate labels")
hyde_template='For each request below, imagine earlier conversations with the same user. Write one or two sentences that the user may have said earlier about themselves that would change how a good assistant answers the request. Reply with a JSON list of objects {{"id": <id>, "note": <text>}}.\n\n{reqs}'
HY=generate(qs,hyde_template,"note","hyde")
HS=st.scores(qs,[[x] for x in HY]); save_method("hyde",sc=HS)
J={}; picks={n:{} for n in ["dense_pick","hyde_pick","property_pick","hyde_property_pick"]}
for protocol in manifest["protocols"]:
    for i,t in enumerate(tasks):
        pool=np.arange(len(lines)) if protocol=="dense365" else np.array(list(range(b["nb"]))+[t["target"]])
        s=S[i]; f=rr(s,pool)+2*rr(B[i],pool)
        # Frozen original pin/gate policy, with all operations restricted to pool.
        f[rank(s,pool)[0]]+=100
        base=rank(f,pool) if gate[i].strip().lower()=="help" else rank(s,pool)
        J[(protocol,i)]=base
        hf=rr(HS[i],pool)+2*rr(B[i],pool)
        hf[rank(HS[i],pool)[0]]+=100
        hb=rank(hf,pool) if gate[i].strip().lower()=="help" else rank(HS[i],pool)
        for name,base0,candidates in [
            ("dense_pick",rank(s,pool),rank(s,pool)[:12]),
            ("hyde_pick",rank(HS[i],pool),rank(HS[i],pool)[:12]),
            ("property_pick",base,list(dict.fromkeys(rank(s,pool)[:6]+base[:6]))),
            ("hyde_property_pick",hb,list(dict.fromkeys(rank(HS[i],pool)[:6]+hb[:6])))]:
            if gate[i].strip().lower()=="recall": order=rank(s,pool)
            else:
                pk=bridge.pick_notes([t["q"]],[[lines[j] for j in candidates]])[0]
                order=rerank(base0,pk,candidates)
            picks[name][(protocol,i)]=order
        if i%5==0: event("rerank",protocol=protocol,done=i+1,total=len(tasks))
save_method("property_gate_pin_rrf",rankings=J)
for name,orders in picks.items(): save_method(name,rankings=orders)
dump("complete.json",dict(finished=time.strftime("%Y-%m-%d %H:%M:%S"),stats=rsi.lm.STATS,formal_test=False))
event("complete")

