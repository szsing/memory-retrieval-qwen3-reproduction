"""Regression check: long-loop SQLite use closes descriptors even with GC disabled."""
from common import *
import tempfile,gc
import night_data as D
from night_budget import cache_connection

def main():
 original=D.DB
 with tempfile.TemporaryDirectory() as folder:
  D.DB=Path(folder)/'test.sqlite3';before=len(list(Path('/proc/self/fd').iterdir()));gc.disable()
  try:
   with D.connect() as c:c.execute('create table checks(value integer)')
   for _ in range(3000):
    with D.connect() as c:assert c.execute('select count(*) from checks').fetchone()[0]==0
    with cache_connection(D.DB) as c:assert c.execute('select count(*) from checks').fetchone()[0]==0
   try:
    with D.connect() as c:c.execute('insert into checks values (1)');raise RuntimeError('rollback fixture')
   except RuntimeError:pass
   with D.connect() as c:assert c.execute('select count(*) from checks').fetchone()[0]==0
   after=len(list(Path('/proc/self/fd').iterdir()));assert after<=before+2,(before,after)
  finally:gc.enable();D.DB=original
 dump(HERE/'night-fd-check.json',{'passed':True,'connection_contexts':6003,'garbage_collection_disabled':True,'descriptors_before':before,'descriptors_after':after,'rollback_preserved':True})
if __name__=='__main__':main()
