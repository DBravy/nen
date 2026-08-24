#!/usr/bin/env python3
import argparse, csv, json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CASES = [
    {"id":"colon_elab","pair_id":"colon1","condition":"positive","group":"colon","target":":","text":"There is one reason the experiment failed: the sensor was unplugged."},
    {"id":"colon_time","pair_id":"colon1","condition":"negative","group":"colon","target":":","text":"The meeting starts at 3:30 tomorrow afternoon."},
    {"id":"who_rel","pair_id":"who1","condition":"positive","group":"who","target":"who","text":"I spoke with Maya, who had already read the report."},
    {"id":"who_meta","pair_id":"who1","condition":"negative","group":"who","target":"who","text":"The word who appears twice in the sentence."},
    {"id":"where_rel","pair_id":"where1","condition":"positive","group":"where","target":"where","text":"We returned to the cabin, where the supplies had been stored."},
    {"id":"where_meta","pair_id":"where1","condition":"negative","group":"where","target":"where","text":"The word where is underlined in the exercise."},
    {"id":"that_comp","pair_id":"that1","condition":"positive","group":"that","target":"that","text":"I realized that the second measurement was wrong."},
    {"id":"that_demo","pair_id":"that1","condition":"negative","group":"that","target":"that","text":"I bought that book at the airport yesterday."},
    {"id":"fact_disc","pair_id":"fact1","condition":"positive","group":"fact","target":"fact","text":"In fact, the second model performed substantially better."},
    {"id":"fact_noun","pair_id":"fact1","condition":"negative","group":"fact","target":"fact","text":"The fact surprised everyone in the room."},
    {"id":"point_temp","pair_id":"point1","condition":"positive","group":"point","target":"point","text":"At one point, I thought the entire analysis had failed."},
    {"id":"point_geom","pair_id":"point1","condition":"negative","group":"point","target":"point","text":"Mark each point on the graph with a small circle."},
    {"id":"feel_pred","pair_id":"feel1","condition":"positive","group":"feel","target":"feel","text":"The lighting makes the room feel much larger than it is."},
    {"id":"feel_noun","pair_id":"feel1","condition":"negative","group":"feel","target":"feel","text":"I like the feel of the paper in this notebook."},
    {"id":"spot_verb","pair_id":"spot1","condition":"positive","group":"spot","target":"spot","text":"If you spot a problem, report it before restarting the machine."},
    {"id":"spot_noun","pair_id":"spot1","condition":"negative","group":"spot","target":"spot","text":"We found a quiet spot beside the river."},
    {"id":"namely","pair_id":None,"condition":"positive","group":"explicit_payload","target":"namely","text":"Only one explanation remained plausible, namely a calibration error."},
    {"id":"for_example","pair_id":None,"condition":"positive","group":"explicit_payload","target":"example","text":"For example, the same pattern appears when the input is reversed."},
    {"id":"the_control","pair_id":None,"condition":"control","group":"generic_incompleteness","target":"the","text":"She quietly opened the wooden box."},
    {"id":"of_control","pair_id":None,"condition":"control","group":"generic_incompleteness","target":"of","text":"He placed a glass of water on the desk."},
    {"id":"to_control","pair_id":None,"condition":"control","group":"generic_incompleteness","target":"to","text":"They decided to postpone the meeting."},
    {"id":"because_boundary","pair_id":None,"condition":"boundary","group":"clausal_opener","target":"because","text":"The experiment stopped because the temperature rose too quickly."},
    {"id":"if_boundary","pair_id":None,"condition":"boundary","group":"clausal_opener","target":"if","text":"The result changes if the final token is removed."},

    {"id":"natural_colon_medicine","pair_id":None,"condition":"natural","group":"natural","target":":","text":"Big news in the world of medicine surfaced this week: The National Institutes of Health promised a whopping $10.1 million to fund the scientific study of ailments."},
    {"id":"natural_fact_skoda","pair_id":None,"condition":"natural","group":"natural","target":"fact","text":"The survey was the biggest car owner survey in the UK. In fact, Skoda took three of the top four spots."},
    {"id":"natural_point_chicken","pair_id":None,"condition":"natural","group":"natural","target":"point","text":"At one point I was wrestling with a whole chicken trying to separate it into the various parts."},
    {"id":"natural_who_crystal","pair_id":None,"condition":"natural","group":"natural","target":"who","text":"I got a call from Crystal, who is the daughter of the owner Ainsley. She told me that the restaurant was in trouble."},
    {"id":"natural_spot_bugs","pair_id":None,"condition":"natural","group":"natural","target":"spot","text":"There is a feedback section in the options in case you spot any more bugs and want to notify the company."},
    {"id":"natural_that_model","pair_id":None,"condition":"natural","group":"natural","target":"that","text":"It may be that your internal model of the world is flawed, and that will deplete your body budget."},
]

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai/gpt-oss-20b")
    p.add_argument("--directions-dir",required=True)
    p.add_argument("--layers",required=True)
    p.add_argument("--sv-indices",required=True,help="zero-based, one per layer")
    p.add_argument("--signs",default=None)
    p.add_argument("--hook-position",choices=["pre","post"],default="post")
    p.add_argument("--out-dir",default="semantic_slot_probe")
    p.add_argument("--trace-left",type=int,default=5)
    p.add_argument("--trace-right",type=int,default=7)
    p.add_argument("--max-length",type=int,default=512)
    return p.parse_args()

def find_file(d, layer):
    files=[x for x in Path(d).rglob("*") if x.suffix.lower() in {".pt",".pth",".npy",".npz"}]
    pats=[rf"layer[_-]?0*{layer}([^0-9]|$)",rf"(^|[^A-Za-z0-9])L0*{layer}([^0-9]|$)"]
    cand=[]
    for x in files:
        s=x.stem
        sc=0
        for i,pat in enumerate(pats):
            if re.search(pat,s,re.I): sc=100-10*i; break
        if sc: cand.append((sc,x))
    if not cand:
        if len(files)==1:return files[0]
        raise FileNotFoundError(f"Could not find direction file for layer {layer}")
    cand.sort(key=lambda z:-z[0]); return cand[0][1]

def load_obj(path):
    if path.suffix in {".pt",".pth"}: return torch.load(path,map_location="cpu",weights_only=False)
    if path.suffix==".npy": return np.load(path,allow_pickle=True)
    if path.suffix==".npz": return dict(np.load(path,allow_pickle=True))
    raise ValueError(path)

def arrays(obj):
    out=[]
    if torch.is_tensor(obj) or isinstance(obj,np.ndarray): return [obj]
    if isinstance(obj,dict):
        pref=["Vh","V","right_singular_vectors","directions","singular_vectors","vectors","svs"]
        for k in pref:
            if k in obj: out += arrays(obj[k])
        for k,v in obj.items():
            if k not in pref: out += arrays(v)
    return out

def extract_vec(obj, sv, d_model, layer):
    if isinstance(obj,dict):
        for k in [layer,str(layer),f"layer_{layer}",f"L{layer}",f"L{layer:02d}"]:
            if k in obj:
                try:return extract_vec(obj[k],sv,d_model,layer)
                except:pass
    for a in arrays(obj):
        t=torch.as_tensor(a)
        if t.ndim==1 and t.numel()==d_model:return t.float()
        if t.ndim==2:
            if t.shape[1]==d_model and sv<t.shape[0]:return t[sv].float()
            if t.shape[0]==d_model and sv<t.shape[1]:return t[:,sv].float()
        if t.ndim==3 and layer<t.shape[0]:
            x=t[layer]
            if x.shape[1]==d_model and sv<x.shape[0]:return x[sv].float()
            if x.shape[0]==d_model and sv<x.shape[1]:return x[:,sv].float()
    raise ValueError(f"Could not extract layer {layer}, SV {sv}, d={d_model}")

def get_blocks(m):
    for fn in [lambda x:x.model.layers,lambda x:x.model.model.layers,lambda x:x.transformer.h,lambda x:x.gpt_neox.layers]:
        try:
            z=fn(m)
            if len(z):return z
        except:pass
    raise RuntimeError("Could not locate decoder blocks")

def occurrences(text,target):
    out=[]; pos=0
    while True:
        i=text.find(target,pos)
        if i<0:return out
        out.append((i,i+len(target))); pos=i+max(1,len(target))

def locate(tok,text,target,maxlen):
    enc=tok(text,return_tensors="pt",return_offsets_mapping=True,truncation=True,max_length=maxlen)
    offs=enc.pop("offset_mapping")[0].tolist()
    occ=occurrences(text,target)
    if not occ: occ=occurrences(text.lower(),target.lower())
    c0,c1=occ[-1]
    ids=[i for i,(a,b) in enumerate(offs) if not(a==b==0) and max(a,c0)<min(b,c1)]
    return enc,ids[-1]

def corr(a,b):
    a=np.array(a,float); b=np.array(b,float)
    if len(a)<3 or a.std()==0 or b.std()==0:return None
    return float(np.corrcoef(a,b)[0,1])

def main():
    a=args()
    layers=[int(x) for x in a.layers.split(",")]
    svs=[int(x) for x in a.sv_indices.split(",")]
    signs=[1]*len(layers) if a.signs is None else [int(x) for x in a.signs.split(",")]
    if not(len(layers)==len(svs)==len(signs)):raise ValueError("layers/svs/signs length mismatch")
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,device_map="auto",torch_dtype="auto")
    model.eval()
    blocks=get_blocks(model); d=model.config.hidden_size
    dirs={}
    for l,sv,sgn in zip(layers,svs,signs):
        f=find_file(a.directions_dir,l)
        v=extract_vec(load_obj(f),sv,d,l).reshape(-1)
        v=sgn*v/v.norm().clamp_min(1e-12)
        dirs[l]=v.cpu()
        print(f"L{l} SV{sv} sign={sgn:+d} <- {f}")

    captured={}
    handles=[]
    for l in layers:
        if a.hook_position=="pre":
            def mk(idx):
                def h(module,inputs): captured[idx]=inputs[0][0].detach().float().cpu()
                return h
            handles.append(blocks[l].register_forward_pre_hook(mk(l)))
        else:
            def mk(idx):
                def h(module,inputs,output):
                    z=output[0] if isinstance(output,(tuple,list)) else output
                    captured[idx]=z[0].detach().float().cpu()
                return h
            handles.append(blocks[l].register_forward_hook(mk(l)))

    indev=model.get_input_embeddings().weight.device
    rows=[]; traces=[]
    with torch.inference_mode():
        for case in CASES:
            enc,idx=locate(tok,case["text"],case["target"],a.max_length)
            input_ids=enc["input_ids"].to(indev)
            mask=enc.get("attention_mask")
            if mask is not None:mask=mask.to(indev)
            captured.clear()
            o=model(input_ids=input_ids,attention_mask=mask,use_cache=False,return_dict=True)
            logits=o.logits[0].detach().float().cpu()
            ids=input_ids[0].detach().cpu()
            lp=torch.log_softmax(logits[idx],dim=-1); pr=lp.exp()
            ent=float(-(pr*lp).sum()); top=float(pr.max())
            nxt=int(ids[idx+1]) if idx+1<len(ids) else None
            surp=float(-lp[nxt]) if nxt is not None else None
            for l,sv in zip(layers,svs):
                H=captured[l]; v=dirs[l]
                raw=H@v; cos=raw/H.norm(dim=-1).clamp_min(1e-12)
                rows.append({
                    **case,"layer":l,"sv_index_zero_based":sv,
                    "activation_raw":float(raw[idx]),"activation_cos":float(cos[idx]),
                    "activation_prev1":float(raw[idx-1]) if idx>0 else None,
                    "activation_next1":float(raw[idx+1]) if idx+1<len(raw) else None,
                    "activation_next2":float(raw[idx+2]) if idx+2<len(raw) else None,
                    "next_token_entropy_nats":ent,"next_token_top_probability":top,
                    "actual_next_surprisal_nats":surp
                })
                lo=max(0,idx-a.trace_left); hi=min(len(raw),idx+a.trace_right+1)
                traces.append({"case_id":case["id"],"layer":l,"trace":[
                    {"rel":j-idx,"token":tok.decode([int(ids[j])]),"raw":float(raw[j]),"cos":float(cos[j])}
                    for j in range(lo,hi)
                ]})

    for h in handles:h.remove()

    summary={"layers":{},"cross_layer":{}}
    for l in layers:
        rr=[r for r in rows if r["layer"]==l]
        pos=[r for r in rr if r["condition"]=="positive"]
        neg=[r for r in rr if r["condition"]=="negative"]
        ctl=[r for r in rr if r["condition"]=="control"]
        nat=[r for r in rr if r["condition"]=="natural"]
        pm=defaultdict(dict)
        for r in rr:
            if r["pair_id"]: pm[r["pair_id"]][r["condition"]]=r
        diffs=[]
        for pid,dct in pm.items():
            if "positive" in dct and "negative" in dct:
                diffs.append({"pair_id":pid,"group":dct["positive"]["group"],
                              "raw_difference":dct["positive"]["activation_raw"]-dct["negative"]["activation_raw"]})
        summary["layers"][str(l)]={
            "positive_mean":float(np.mean([r["activation_raw"] for r in pos])),
            "negative_mean":float(np.mean([r["activation_raw"] for r in neg])),
            "control_mean":float(np.mean([r["activation_raw"] for r in ctl])),
            "natural_mean":float(np.mean([r["activation_raw"] for r in nat])),
            "mean_pair_difference":float(np.mean([x["raw_difference"] for x in diffs])),
            "pair_win_fraction":float(np.mean([x["raw_difference"]>0 for x in diffs])),
            "activation_entropy_r":corr([r["activation_raw"] for r in rr],[r["next_token_entropy_nats"] for r in rr]),
            "activation_surprisal_r":corr([r["activation_raw"] for r in rr],[r["actual_next_surprisal_nats"] for r in rr]),
            "pair_effects":diffs
        }
    if len(layers)>=2:
        for i in range(len(layers)):
            for j in range(i+1,len(layers)):
                a1,a2=layers[i],layers[j]
                d1={r["id"]:r for r in rows if r["layer"]==a1}
                d2={r["id"]:r for r in rows if r["layer"]==a2}
                ks=sorted(set(d1)&set(d2))
                summary["cross_layer"][f"{a1}_vs_{a2}"]={
                    "activation_r":corr([d1[k]["activation_raw"] for k in ks],[d2[k]["activation_raw"] for k in ks])
                }

    with open(out/"results.json","w") as f: json.dump({"summary":summary,"items":rows,"traces":traces},f,indent=2)
    with open(out/"items.csv","w",newline="") as f:
        keys=list(rows[0].keys()); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(out/"summary.md","w") as f:
        f.write("# Semantic slot probe\n\n")
        f.write("Hypothesis: the direction marks prospective semantic payload / slot opening.\n\n")
        for l,s in summary["layers"].items():
            f.write(f"## Layer {l}\n")
            for k,v in s.items():
                if k!="pair_effects":f.write(f"- {k}: {v}\n")
            f.write("\n")
    print(f"Wrote {out/'results.json'}, {out/'items.csv'}, {out/'summary.md'}")

if __name__=="__main__":
    main()
