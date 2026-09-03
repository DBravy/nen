"""Score an LLM's blind labels of blind_tails.md against the hidden key.

Usage:  python score_blind_judgments.py labels.tsv
labels.tsv: one line per item, `id<TAB>LABEL` (LABEL in REFERENCE/STANCE/MIXED/STRUCTURAL/OTHER).
Prints, per bank and per tier, the distribution of labels and a stance-vs-reference odds ratio with
a Fisher exact test (scipy if available). Also writes labels_by_bank.tsv joining labels to the key.
"""
import sys, csv, collections, math, os
here=os.path.dirname(os.path.abspath(__file__))
key={}
with open(os.path.join(here,"blind_tails_key.tsv")) as f:
    for row in csv.DictReader(f,delimiter="\t"): key[row["id"]]=row
labels={}
with open(sys.argv[1]) as f:
    for line in f:
        parts=line.strip().split("\t")
        if len(parts)>=2 and parts[0] in key: labels[parts[0]]=parts[1].strip().upper()
print(f"labelled {len(labels)} of {len(key)} items")
def table(rows,name):
    c=collections.defaultdict(collections.Counter)
    for i,l in labels.items():
        if rows(key[i]): c[key[i]["bank"]][l]+=1
    print(f"\n## {name}")
    labs=["REFERENCE","STANCE","MIXED","STRUCTURAL","OTHER"]
    print("bank      "+"".join(f"{l:>11s}" for l in labs)+"   S/(S+R)")
    for b in ("UWS","GSS"):
        n=c[b]; s,r=n["STANCE"],n["REFERENCE"]
        print(f"{b:9s} "+"".join(f"{n[l]:11d}" for l in labs)+f"   {s/(s+r) if s+r else float('nan'):.2f}")
    a,b_=c["UWS"]["STANCE"],c["UWS"]["REFERENCE"]; cc,d=c["GSS"]["STANCE"],c["GSS"]["REFERENCE"]
    if min(a,b_,cc,d)>=0:
        try:
            from scipy.stats import fisher_exact
            odds,p=fisher_exact([[a,b_],[cc,d]]); print(f"UWS-vs-GSS stance:reference odds ratio = {odds:.2f}, Fisher p = {p:.3g}")
        except Exception as e:
            print("(scipy not available for Fisher test)")
table(lambda r:True,"ALL items")
table(lambda r:r["tier"]=="selective_top100","selective_top100 (selected polarity)")
table(lambda r:r["tier"]=="broad_top15","broad_top15 (both polarities)")
table(lambda r:r["tier"]=="mid_random","mid_random")
with open(os.path.join(here,"labels_by_bank.tsv"),"w") as f:
    f.write("id\tbank\tcandidate\tpolarity\ttier\tlabel\n")
    for i,l in labels.items(): k=key[i]; f.write(f"{i}\t{k['bank']}\t{k['candidate']}\t{k['polarity']}\t{k['tier']}\t{l}\n")
print("\nwrote labels_by_bank.tsv")
