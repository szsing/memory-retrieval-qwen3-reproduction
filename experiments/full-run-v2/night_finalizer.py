"""Finish reports after the current supervisor exits, including a budget pause."""
from night_data import *
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True);a=p.parse_args()
 proc=Path('/proc')/str(a.parent)/'stat'
 original=proc.read_text().split()[21] if proc.exists() else None
 while proc.exists():
  parts=proc.read_text().split()
  if parts[21]!=original or parts[2]=='Z':break
  time.sleep(30)
 from night_status import status
 from night_analysis import main
 summary=status();main();dump(HERE/'night-finished.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'supervisor_pid':a.parent,'meaning':'queue exited; inspect complete/paused/failed coverage, not an assertion all benchmarks finished','datasets':summary['datasets']})
