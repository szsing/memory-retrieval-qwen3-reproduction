"""Format-only constraints and strict indexing; no semantic rewriting or missing-item defaults."""
import json,re
class InvalidOutput(ValueError):pass

def contract(prompt):
 if prompt.startswith('Notes saved about the user from earlier chats'):return None
 if prompt.startswith('For each item, a hidden fact about the user is given'):
  ids=[int(x) for x in re.findall(r'(?m)^Item (\d+)\s*$',prompt)];kind='judge';field='v';value={'enum':['yes','no']}
 elif prompt.startswith('A user asks an assistant:'):
  n=max([int(x) for x in re.findall(r'(?m)^(\d+)\. ',prompt)],default=0)
  if not n:raise InvalidOutput('Picker has no numbered candidates')
  return {'kind':'picker','schema':{'type':'object','properties':{'pick':{'type':'array','items':{'type':'integer','minimum':1,'maximum':n},'maxItems':6}},'required':['pick'],'additionalProperties':False}}
 else:
  ids=[int(x) for x in re.findall(r'(?m)^ID (\d+):',prompt)]
  if not ids:return None
  head=re.split(r'\n(?:LINES|REQUESTS):',prompt,maxsplit=1)[0]
  if '"p"' in head:kind='properties';field='p';value={'type':'array','items':{'type':'string'},'maxItems':8}
  elif '"k"' in head:kind='gate';field='k';value={'enum':['help','recall']}
  elif '"note"' in head:kind='hyde';field='note';value={'type':'string'}
  else:raise InvalidOutput('Unrecognized numbered JSON prompt; refuse guessing')
 if not ids or len(ids)!=len(set(ids)):raise InvalidOutput('Empty/duplicate prompt IDs')
 entries=[{'type':'object','properties':{'id':{'type':'integer','const':i},field:value},'required':['id',field],'additionalProperties':False} for i in ids]
 return {'kind':kind,'ids':ids,'field':field,'schema':{'type':'array','prefixItems':entries,'items':False,'minItems':len(ids),'maxItems':len(ids)}}

def decode(raw):
 text=raw.strip()
 if text.startswith('```'):
  m=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',text,re.S|re.I)
  if not m:raise InvalidOutput('Incomplete code fence')
  text=m[1].strip()
 try:return json.loads(text)
 except json.JSONDecodeError:
  # Accept complete concatenated JSON objects only; never repair quoted numbers or alter text.
  decoder=json.JSONDecoder();values=[];pos=0
  while pos<len(text):
   while pos<len(text) and text[pos].isspace():pos+=1
   if pos==len(text):break
   try:v,end=decoder.raw_decode(text,pos)
   except json.JSONDecodeError as e:raise InvalidOutput('Not valid complete JSON/NDJSON') from e
   values.append(v);pos=end
  if not values or not all(isinstance(v,dict) for v in values):raise InvalidOutput('Ambiguous multi-document JSON')
  return values

def normalize(raw,c):
 import jsonschema
 obj=decode(raw);changes=[]
 if c['kind']=='picker':jsonschema.validate(obj,c['schema']);return obj,changes
 if isinstance(obj,dict):
  if set(obj)=={'items'} and isinstance(obj['items'],list):obj=obj['items'];changes.append('unwrap_items')
  elif set(obj)=={'id',c['field']}:obj=[obj];changes.append('wrap_single_item')
  else:raise InvalidOutput('Unexpected object container')
 if not isinstance(obj,list):raise InvalidOutput('Expected array')
 obj=[dict(x) if isinstance(x,dict) else x for x in obj]
 for x in obj:
  if not isinstance(x,dict):raise InvalidOutput('Non-object array item')
  if isinstance(x.get('id'),str) and re.fullmatch(r'0|[1-9][0-9]*',x['id']):x['id']=int(x['id']);changes.append('decimal_string_id')
 ids=[x.get('id') for x in obj]
 if any(type(i) is not int for i in ids) or len(set(ids))!=len(ids):raise InvalidOutput('Bad/duplicate IDs')
 # Original answer prompt numbers Item1..N, while by_id consumes positions0..N-1.
 # Legacy 0-based judge output is accepted only when the entire bijection is present.
 if c['kind']=='judge' and set(ids)==set(range(len(c['ids']))):
  for x in obj:x['id']+=1
  ids=[x['id'] for x in obj];changes.append('complete_zero_based_judge_to_prompt_ids')
 if set(ids)!=set(c['ids']) or len(ids)!=len(c['ids']):raise InvalidOutput('Missing or extra IDs; no imputation')
 byid={x['id']:x for x in obj};obj=[byid[i] for i in c['ids']]
 jsonschema.validate(obj,c['schema']);return obj,changes

def consumer_value(obj,c):
 if c and c['kind']=='judge':return [{**x,'id':x['id']-1} for x in obj]
 return obj
