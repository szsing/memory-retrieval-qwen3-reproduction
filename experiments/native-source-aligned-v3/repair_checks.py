import os,sys,json,hashlib,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];os.environ['MEM_PROJECT_ROOT']=str(root);os.environ['NATIVE_DEADLINE']=(p/'deadline.txt').read_text().strip();os.environ['NATIVE_JOB']='repair-storage-isolation'
sys.path.insert(0,str(p));import native
from transport import dump
m1=native.mem0(p/'checks/storage-a');m2=native.mem0(p/'checks/storage-b')
a=str(m1._telemetry_vector_store.client._client.location);b=str(m2._telemetry_vector_store.client._client.location)
assert a!=b,(a,b)
native.close_memory(m1,'mem0');native.close_memory(m2,'mem0')
dump(p/'checks/storage-isolation.json',{'passed':True,'two_concurrent_instances_opened':True,'distinct_sdk_storage':[a,b]})
print('STORAGE_ISOLATION_PASSED',flush=True)
os.environ['NATIVE_JOB']='root-native-json-object-probe'
for line in (p/'api-responses.jsonl').open():
 x=json.loads(line)
 if x['job']=='retrieve-inmind-rootmem-' and x['response']['choices'][0]['finish_reason']=='length':body=x['request'];break
v=native.T.call(body['messages'],max_tokens=8192,response_format={'type':'json_object'},json_output=True)
import jsonschema
jsonschema.validate(v,native.ROOT_SCHEMA)
dump(p/'checks/root-native-json-object.json',{'passed':True,'skills':len(v['agent_skills']),'matrix':v,'messages_sha256':hashlib.sha256(json.dumps(body['messages']).encode()).hexdigest(),'temperature':0,'max_tokens':8192,'response_format':'native json_object'})
print('ROOT_NATIVE_JSON_OBJECT_PASSED',len(v['agent_skills']),flush=True)
