"""S must have complete retrieval AND evaluation before B can enter the queue."""
from night_data import *
from night_budget import check_budget,NightBudgetExpired
import subprocess,shutil

def main():
 try:check_budget()
 except NightBudgetExpired:return
 with connect() as c:
  remaining=c.execute("select count(*) from jobs where dataset='lucid_s' and status!='complete'").fetchone()[0]
  rows=c.execute("select value from results where dataset='lucid_s'").fetchall()
  ready=not remaining and len(rows)==1936*15 and all('evaluation' in json.loads(v) for (v,) in rows)
 if not ready:
  dump(HERE/'lucid-b-extension-status.json',{'status':'waiting_for_full_S','S_only_first':True});return
 if shutil.disk_usage(HERE).free<10*1024**3:
  dump(HERE/'lucid-b-extension-status.json',{'status':'waiting_for_disk','minimum_free_gib':10});return
 subprocess.run([sys.executable,str(HERE/'download_full_data.py'),'--lucid-b'],check=True)
 check_budget();append_lucid_b()
 dump(HERE/'lucid-b-extension-status.json',{'status':'data_ready'});
 # The supervisor will seed and launch the same full method matrix on the new B cases.
 (HERE/'lucid-b-ready').write_text('verified and normalized\n')
if __name__=='__main__':main()
