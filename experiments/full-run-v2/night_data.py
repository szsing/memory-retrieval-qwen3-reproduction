"""Lossless input normalization. Gold metadata never enters method inputs."""
from common import *
import collections
from contextlib import contextmanager
DB=HERE/'night.sqlite3'
@contextmanager
def connect():
 c=sqlite3.connect(DB,timeout=120)
 try:
  c.execute('pragma journal_mode=WAL')
  with c:yield c
 finally:c.close()
def init():
 with connect() as c:
  c.executescript('''create table if not exists notes(id text primary key, messages text, text text, user_text text);
create table if not exists cases(dataset text,id text,history text,queries text,primary key(dataset,id));
create table if not exists features(key text primary key,value text);
create table if not exists jobs(dataset text,id text,method text,status text default 'pending',error text,updated real, primary key(dataset,id,method));
create table if not exists results(dataset text,id text,method text,qid text,value text,primary key(dataset,id,method,qid));''')
def add_case(c,dataset,cid,sessions,queries):
 history=[]
 for sid,turns in sessions:
  clean=[{'role':m['role'],'content':m['content']} for m in turns]
  assert all(isinstance(m['content'],str) and m['role'] in ('user','assistant','system') for m in clean)
  chunks=pairs({'turns':clean})
  assert [m for ch in chunks for m in ch]==clean,'lossy chunking'
  for ch in chunks:
   raw=json.dumps(ch,ensure_ascii=False,sort_keys=True);nid=hashlib.sha256(raw.encode()).hexdigest()
   c.execute('insert or ignore into notes values (?,?,?,?)',(nid,raw,text(ch),'\n'.join(m['content'] for m in ch if m['role']=='user')))
   history.append({'note':nid,'session':str(sid)})
 assert history
 assert len({q['id'] for q in queries})==len(queries)
 for q in queries:
  assert isinstance(q['text'],str) and q['text'].strip()
  assert set(q['gold_sessions'])<=set(str(s) for s,_ in sessions),(dataset,cid,q['gold_sessions'])
 c.execute('insert into cases values (?,?,?,?)',(dataset,str(cid),json.dumps(history),json.dumps(queries)))
def normalize():
 init()
 with connect() as c:
  if c.execute('select count(*) from cases').fetchone()[0]:
   assert (HERE/'data-audit.json').exists(),'Normalization interrupted; remove partial night DB before rebuilding';return
  tasks,_,_=load_data();dev=set(dev_ids(tasks))
  for t in tasks:
   sessions=timeline(t);gs=[str(s['session_id']) for s in sessions if any(t['user_message']==m['content'] for m in s['turns'])]
   add_case(c,'inmind',t['task_id'],[(s['session_id'],s['turns']) for s in sessions],[{'id':k,'text':t[k],'gold_sessions':gs,'gold':t,'heldout':t['task_id'] not in dev} for k in ['query','naive_query']])
  lucid=json.loads((HERE/'data/lucid_s.json').read_text());assert len(lucid)==1936
  for ri,r in enumerate(lucid):
   assert len(r['haystack_sessions'])==len(r['haystack_session_ids'])==50
   gold={k:v for k,v in r.items() if k not in ('haystack_sessions','haystack_session_ids')}
   add_case(c,'lucid_s',str(ri)+'__'+r['query_id'],list(zip(r['haystack_session_ids'],r['haystack_sessions'])),[{'id':r['query_id'],'text':r['query'],'gold_sessions':list(map(str,r['answer_session_ids'])),'gold':gold}])
  import pyarrow.parquet as pq
  rows=[r for p in sorted((HERE/'data').glob('implex-*.parquet')) for r in pq.read_table(p).to_pylist()];assert len(rows)==1550
  empty=0;prefix_sessions=0;unlabeled_sessions=0;empty_turns=0
  for i,r in enumerate(rows):
   sessions=[]
   for sid,s in r['conversation'].items():
    if s is None or not s.strip():continue
    ms=list(re.finditer(r'(?m)^(Speaker\s*1|Assistant):[ \t]*',s))
    prefix=s[:ms[0].start()] if ms else s
    turns=[]
    if prefix.strip():
     turns.append({'role':'assistant','content':prefix.strip()});prefix_sessions+=1
     if not ms:unlabeled_sessions+=1
    turns += [{'role':'user' if m[1].replace(' ','')=='Speaker1' else 'assistant','content':s[m.end():ms[j+1].start() if j+1<len(ms) else len(s)].strip()} for j,m in enumerate(ms)]
    empty_turns+=sum(not m['content'] for m in turns)
    sessions.append((str(sid),turns))
   if not r['qa']:empty+=1
   add_case(c,'implex',i,sessions,[{'id':str(j),'text':q['question'],'gold_sessions':list(map(str,q['retrieved_conv_ids'])),'gold':q} for j,q in enumerate(r['qa'])])
  audit={'datasets':{},'notes':c.execute('select count(*) from notes').fetchone()[0],'implex_empty_qa_records':empty,'implex_empty_turns_preserved':empty_turns,'implex_prefixed_sessions':prefix_sessions,'implex_unlabeled_sessions':unlabeled_sessions,'implex_parse_policy':'Speaker1/Speaker 1 map to user; Assistant to assistant; unlabelled generator prose retained as assistant text and excluded from user-only property extraction','input_policy':'all source sessions and all roles preserved; interaction chunks; no gold fields in notes; no history truncation','lucid_unique_query_ids':len({r['query_id'] for r in lucid}),'lucid_row_identity':'row index + query_id; duplicated IDs retained, never silently deduplicated','local_model':MODEL,'controlled_adaptation':True}
  for ds in ['inmind','lucid_s','implex']:
   rr=c.execute('select history,queries from cases where dataset=?',(ds,)).fetchall()
   audit['datasets'][ds]={'records':len(rr),'questions':sum(len(json.loads(q)) for _,q in rr),'interaction_chunks_sum':sum(len(json.loads(h)) for h,_ in rr),'session_counts':dict(collections.Counter(len(set(x['session'] for x in json.loads(h))) for h,_ in rr))}
  dump(HERE/'data-audit.json',audit)
 event('data_audit_complete',**audit)
def append_lucid_b():
 with connect() as c:
  if c.execute("select count(*) from cases where dataset='lucid_b'").fetchone()[0]:return
  rows=json.loads((HERE/'data/lucid_b.json').read_text());assert len(rows)==1936
  for ri,r in enumerate(rows):
   assert len(r['haystack_sessions'])==len(r['haystack_session_ids'])==200
   gold={k:v for k,v in r.items() if k not in ('haystack_sessions','haystack_session_ids')}
   add_case(c,'lucid_b',str(ri)+'__'+r['query_id'],list(zip(r['haystack_session_ids'],r['haystack_sessions'])),[{'id':r['query_id'],'text':r['query'],'gold_sessions':list(map(str,r['answer_session_ids'])),'gold':gold}])
 audit=json.loads((HERE/'data-audit.json').read_text());audit['datasets']['lucid_b']={'records':1936,'questions':1936,'sessions_each':200};dump(HERE/'data-audit.json',audit)
if __name__=='__main__':
 if '--lucid-b' in sys.argv:append_lucid_b()
 else:normalize()
