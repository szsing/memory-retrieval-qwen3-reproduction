"""Run unchanged retrieval with a read-only shared embedding-cache snapshot."""
import sys,runpy
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];V3=ROOT/'experiments/native-source-aligned-v3';SRC=ROOT/'experiments/original-qwen-format-v2/snapshot/rsi-memory'
sys.path.insert(0,str(SRC));import ret
ret.CDIR=str(OUT/'embedding-cache-readonly');ret.save=lambda:None
runpy.run_path(str(V3/'worker.py'),run_name='__main__')

