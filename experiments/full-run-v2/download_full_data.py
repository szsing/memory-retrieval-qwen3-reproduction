"""Pinned, ranged, checksum-verified full benchmark files; no row sampling."""
from common import *
from concurrent.futures import ThreadPoolExecutor
import urllib.error
DATA=HERE/'data';DATA.mkdir(exist_ok=True)
SPECS=[
 ('implex-0.parquet','Kaylee0501/ImplexConv-opposed','main','data/opposed-00000-of-00002.parquet','7745febed883cb045c251fccd17dfe476de732965dd381c2e63d6bf51c3c94ce'),
 ('implex-1.parquet','Kaylee0501/ImplexConv-opposed','main','data/opposed-00001-of-00002.parquet','6bf51cd8a216620a71d8bc27cd3b67030f02ed0ebf2b7eb7adce9f9316268dee'),
 ('lucid_s.json','MichiganNLP/LUCid','2431979d0f9e8ec9688a33cf21a77f6c967e5a9c','lucid_s.json','d62ceb39e18544403a00bb68f0ecee0378f1741bcddc37cc801dbd79504fd4ef')]
if '--lucid-b' in sys.argv:
 SPECS.append(('lucid_b.json','MichiganNLP/LUCid','2431979d0f9e8ec9688a33cf21a77f6c967e5a9c','lucid_b.json','ed1d088c395064e58d8c55438174fe696a4bdd56c323df37468dbc98d40f5836'))
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args):return None
opener=urllib.request.build_opener(NoRedirect())
manifest={}
for filename,repo,revision,remote,checksum in SPECS:
 assert len(checksum)==64
 url='https://hf-mirror.com/datasets/'+repo+'/resolve/'+revision+'/'+remote
 for attempt in range(8):
  try:
   try:h=opener.open(urllib.request.Request(url+'?download=true&metadata='+str(time.time_ns()),method='HEAD'),timeout=45).headers
   except urllib.error.HTTPError as e:
    if e.code not in (301,302,303,307,308):raise
    h=e.headers
   size=int(h['x-linked-size']);revision=h.get('x-repo-commit',revision);etag=h.get('x-linked-etag','').strip('"')
   assert etag==checksum,(filename,'unexpected official LFS hash',etag)
   break
  except Exception as e:
   event('download_metadata_retry',file=filename,attempt=attempt,error=str(e));time.sleep(2)
 else:raise RuntimeError('metadata unavailable '+filename)
 url='https://hf-mirror.com/datasets/'+repo+'/resolve/'+revision+'/'+remote
 manifest[filename]={'dataset':repo,'revision':revision,'file':remote,'bytes':size,'sha256':checksum,'verified':False}
 dump(DATA/'download-manifest.json',manifest)
 final=DATA/filename
 if final.exists() and sha(final)==checksum:
  manifest[filename]['verified']=True;dump(DATA/'download-manifest.json',manifest);continue
 parts=DATA/(filename+'.parts');parts.mkdir(exist_ok=True);chunk=8*1024*1024
 def get(start):
  end=min(size,start+chunk)-1;length=end-start+1;path=parts/str(start)
  if path.exists() and path.stat().st_size==length:return
  for attempt in range(12):
   try:
    req=urllib.request.Request(url+'?download=true&piece='+str(start)+'&attempt='+str(time.time_ns()),headers={'Range':'bytes='+str(start)+'-'+str(end)})
    with urllib.request.urlopen(req,timeout=180) as response:
     assert response.status==206 and response.headers.get('Content-Range','').startswith('bytes '+str(start)+'-'+str(end)+'/'),(response.status,dict(response.headers))
     content=response.read(length+1)
    assert len(content)==length
    temp=path.with_suffix('.tmp');temp.write_bytes(content);temp.replace(path)
    return
   except Exception as e:
    event('download_piece_retry',file=filename,start=start,attempt=attempt,error=str(e));time.sleep(2)
  raise RuntimeError('range failed '+filename+' '+str(start))
 with ThreadPoolExecutor(max_workers=8) as pool:
  for done,_ in enumerate(pool.map(get,range(0,size,chunk)),1):event('download_progress',file=filename,done=done,total=(size+chunk-1)//chunk)
 temporary=DATA/(filename+'.tmp')
 with temporary.open('wb') as writer:
  for start in range(0,size,chunk):writer.write((parts/str(start)).read_bytes())
 assert temporary.stat().st_size==size and sha(temporary)==checksum,'full file SHA-256 mismatch '+filename
 temporary.replace(final);manifest[filename]['verified']=True;dump(DATA/'download-manifest.json',manifest)
 for part in parts.iterdir():part.unlink()
 parts.rmdir();event('full_data_file_ready',file=filename,bytes=size)
dump(DATA/'all-files-ready.json',manifest)
