"""Common reader and explicit local judge adaptation; no gold in generation."""
from common import *
import types
from night_budget import NightTransport
T=NightTransport('night-reader-evaluator')
SCORE={'type':'object','properties':{'score':{'type':'integer','enum':[0,1]},'reason':{'type':'string'}},'required':['score','reason']}
def evaluate(dataset,q,context):
 if dataset=='inmind':
  prompts=VENDOR/'InMind-main/evaluation/prompts';g=q['gold']
  ans=T.call([{'role':'system','content':(prompts/'answer_system.txt').read_text().replace('{context}',context)},{'role':'user','content':q['text']}],max_tokens=2048)
  payload={k:g[k] for k in ['user_message','explanation']};payload.update(context=context,query=q['text'],answer=ans)
  metric='judge_application.txt' if q['id']=='query' else 'judge_naive.txt'
  score=T.call([{'role':'system','content':(prompts/metric).read_text()},{'role':'user','content':json.dumps(payload)}],schema=SCORE,json_output=True)
  payload.pop('answer');recall=T.call([{'role':'system','content':(prompts/'judge_target_recall.txt').read_text()},{'role':'user','content':json.dumps(payload)}],schema=SCORE,json_output=True)
  return {'answer':ans,'score':score['score'],'judge':score,'target_recall':recall,'heldout':q['heldout'],'protocol':'original InMind prompts, local reader/judge adaptation'}
 prompt=f'Please answer the question.\n\nHistory Chats:\n\n{context}\nQuestion: {q["text"]}\nAnswer:'
 ans=T.call([{'role':'user','content':prompt}],max_tokens=2048)
 if dataset.startswith('lucid_'):
  # Original official scoring dispatcher/prompts/options, only provider replaced.
  sys.path.insert(0,str(VENDOR/'LUCid-main'))
  if 'src.llm_client' not in sys.modules:
   stub=types.ModuleType('src.llm_client')
   def local_query(model_path,user_prompt,temperature,max_tokens,print_tokens):
    schema={'type':'object','properties':{'evaluation':{'type':'string'},'reasoning':{'type':'string'}},'required':['evaluation','reasoning']}
    v=T.call([{'role':'user','content':user_prompt}],max_tokens=max(1024,max_tokens),schema=schema,json_output=True)
    return json.dumps(v),None,None
   stub.query_llm=local_query;sys.modules['src.llm_client']=stub
  from src.evaluation.runner import evaluate_entry
  g=copy_gold=dict(q['gold']);g['hypothesis']=ans
  judged=evaluate_entry(g,model_name=MODEL)
  if not str(judged.get('evaluator_prediction_letter',judged.get('evaluator_prediction',''))).strip():raise ValueError('Empty LUCid judge prediction')
  return {'answer':ans,'score':judged['evaluator_match'],'judge':judged,'protocol':'official LUCid evaluator with local Qwen replacement; not official model leaderboard'}
 g=q['gold']
 # Extract the official judge verbatim without importing Groq or its retry loop.
 import ast
 source=(VENDOR/'ImplexConv-main/evaluate/get_baseline_scores.py').read_text()
 tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_llm_judge')
 assign=fn.body[0]
 ns={'question':q['text'],'response':ans,'reason':g['opposed_implicit_reasoning']}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[assign],type_ignores=[])),'official_implex_judge_prompt','exec'),ns)
 raw=T.call([{'role':'user','content':ns['evaluation_prompt']}],max_tokens=2048)
 m=re.match(r'^\s*([01])(?:\b|[.:])',raw)
 if not m:raise ValueError('Official ImplexConv judge did not start with 0/1: '+raw[:120])
 from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
 from rouge import Rouge
 ref=str(g['answer']);smooth=SmoothingFunction().method1
 lexical={f'bleu_{j}':sentence_bleu([ref.split()],ans.split(),weights=w,smoothing_function=smooth) for j,w in [(1,(1,0,0)),(2,(0.5,0.5,0)),(3,(0.33,0.33,0.33))]}
 lexical['rouge_l']=Rouge().get_scores(ans,ref)[0]['rouge-l']['f'] if ans.strip() and ref.strip() else 0
 return {'answer':ans,'score':int(m[1]),'judge_raw':raw,'lexical':lexical,'protocol':'official ImplexConv binary judge prompt and BLEU/ROUGE formulas; common local reader and local judge adaptation'}
