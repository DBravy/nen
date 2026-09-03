"""Score tails in both banks using ANY word-level labelling (yours, mine, or an LLM's).

Usage:  python score_word_judgments.py word_labels.tsv
word_labels.tsv: lines `word<TAB>LABEL` with LABEL in R/S/F/X (case-insensitive; also accepts the
long names REFERENCE/STANCE/FUNCTION/FRAGMENT). Words are matched case-insensitively; unlabelled
words are ignored. For each tail the stance ratio is S/(R+S) over its whole-word centre tokens; tails
with fewer than 10 labelled R+S words are dropped. Prints per-bank comparisons at several tiers with
a Mann-Whitney test and a Fisher test on stance- vs reference-dominant tails, and writes
tail_scores_from_<labelsfile>.csv.
"""
import sys, json, os, csv, math
from collections import Counter
here=os.path.dirname(os.path.abspath(__file__))
labs={}
with open(sys.argv[1]) as f:
    for line in f:
        p=line.rstrip("\n").split("\t")
        if len(p)>=2 and p[0].strip():
            l=p[1].strip().upper()[:1]
            if l in "RSFX": labs[p[0].strip().lower()]=l
print(f"loaded {len(labs)} word labels")
rows=[]
for line in open(os.path.join(here,"tail_words.jsonl")):
    t=json.loads(line); c=Counter(labs.get(w.lower(),"U") for w in t["words"])
    R,S=c["R"],c["S"]; t.update(R=R,S=S,F=c["F"],X=c["X"],U=c["U"],ratio=(S/(R+S) if R+S else float("nan"))); rows.append(t)
try:
    from scipy.stats import mannwhitneyu, fisher_exact
except Exception:
    mannwhitneyu=fisher_exact=None
def compare(name,keep,MIN=10):
    a=[r for r in rows if r["bank"]=="UWS" and keep(r) and r["R"]+r["S"]>=MIN]
    b=[r for r in rows if r["bank"]=="GSS" and keep(r) and r["R"]+r["S"]>=MIN]
    if len(a)<5 or len(b)<5: print(f"{name}: too few tails"); return
    ra=[r["ratio"] for r in a]; rb=[r["ratio"] for r in b]
    mean=lambda x:sum(x)/len(x); med=lambda x:sorted(x)[len(x)//2]
    sa=sum(x>=.6 for x in ra); sb=sum(x>=.6 for x in rb); da=sum(x<=.4 for x in ra); db=sum(x<=.4 for x in rb)
    ew=lambda L:sum(r["S"] for r in L)/max(1,sum(r["S"]+r["R"] for r in L))
    line=f"{name:52s} n={len(a)}/{len(b)}  mean ratio {mean(ra):.3f}/{mean(rb):.3f}  median {med(ra):.3f}/{med(rb):.3f}  stance-dom {sa}/{sb} ref-dom {da}/{db}  event-weighted {ew(a):.3f}/{ew(b):.3f}"
    if mannwhitneyu:
        p=mannwhitneyu(ra,rb,alternative="two-sided").pvalue; odds,pf=fisher_exact([[sa,da],[sb,db]])
        line+=f"  MWU p={p:.2g}  Fisher OR={odds:.2f} p={pf:.2g}"
    print(line)
compare("ALL tails (both polarities)",lambda r:True)
compare("ALL selected-polarity tails",lambda r:r["selected"])
for N in (40,100,300,600): compare(f"top-{N} selective (selected polarity)",lambda r,N=N:r["selected"] and r["selrank"]<=N)
for N in (60,150):
    tops={bk:set(sorted({r["candidate"]:r["cos"] for r in rows if r["bank"]==bk}.items(),key=lambda kv:-kv[1])[:N]) for bk in ("UWS","GSS")}
    tops={bk:{c for c,_ in v} for bk,v in tops.items()}
    compare(f"top-{N} broad (both polarities)",lambda r,tops=tops:r["candidate"] in tops[r["bank"]])
out=os.path.join(here,"tail_scores_from_"+os.path.basename(sys.argv[1]).rsplit(".",1)[0]+".csv")
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["bank","candidate","polarity","selected","selrank","layer","R","S","F","X","U","ratio"])
    for r in rows: w.writerow([r["bank"],r["candidate"],r["polarity"],r["selected"],r["selrank"],r["layer"],r["R"],r["S"],r["F"],r["X"],r["U"],f"{r['ratio']:.3f}" if r["ratio"]==r["ratio"] else ""])
print("wrote",out)
