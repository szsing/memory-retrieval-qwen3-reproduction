from night_data import *
from night_methods import props
from concurrent.futures import ThreadPoolExecutor
p=HERE/'controlled_constraint_donors.json';d=json.loads(p.read_text());items=[(g,t) for g,ts in d.items() for t in ts[:3]]
with ThreadPoolExecutor(max_workers=4) as ex:
 out=list(ex.map(lambda x:{'group':x[0],'fact':x[1],'properties':props(x[1],True)},items))
dump(HERE/'controlled-constraint-smoke.json',out)
for r in out:print(json.dumps(r))
