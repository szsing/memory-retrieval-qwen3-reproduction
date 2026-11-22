from night_data import *
from night_methods import props
from concurrent.futures import ThreadPoolExecutor
p=HERE/'controlled_manual_donors.json';d=json.loads(p.read_text());items=[(g,t) for g,ts in d.items() for t in ts]
with ThreadPoolExecutor(max_workers=4) as ex:
 out=list(ex.map(lambda x:{'group':x[0],'fact':x[1],'properties':props(x[1],True)},items))
dump(HERE/'controlled-manual-property-audit.json',out)
print({g:sum(bool(r['properties']) for r in out if r['group']==g) for g in d})
