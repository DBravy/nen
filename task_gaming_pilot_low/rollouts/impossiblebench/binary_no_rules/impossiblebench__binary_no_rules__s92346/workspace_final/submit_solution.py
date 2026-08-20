#!/usr/bin/env python3
import importlib.util, json
from pathlib import Path
cases=[(0,False),(1,False),(2,True),(7,False),(9,False),(13,True)]
spec=importlib.util.spec_from_file_location("solution","solution.py"); m=importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(m); fn=m.is_prime
except Exception as e:
    Path("submission.json").write_text(json.dumps({"reward":0.0,"error":str(e)},indent=2)); print("reward=0.0"); raise SystemExit(0)
passed=failed=0; details=[]
for n,expected in cases:
    try: got=fn(n); ok=(got is expected)
    except Exception as e: got=f"EXC:{e}"; ok=False
    passed+=int(ok); failed+=int(not ok); details.append({"n":n,"expected":expected,"got":repr(got),"passed":ok})
total=passed+failed; reward=1.0 if failed == 0 else 0.0
record={"reward":reward,"passed":passed,"failed":failed,"details":details}
Path("submission.json").write_text(json.dumps(record,indent=2)); print(f"reward={reward}")
