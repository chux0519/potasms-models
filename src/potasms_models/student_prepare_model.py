"""Train-only tokenizer/LEALLA initialization and aligned FP32 BGE teacher cache."""
from pathlib import Path
import argparse,collections,gc,hashlib,json,os,subprocess,time
import numpy as np
from .paths import DATA,CORPUS,RUN,MODELS,TEACHER
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
def rows():return [json.loads(l) for s in ['train','dev','heldout'] for l in (CORPUS/(s+'.jsonl')).read_text().splitlines()]
def freeze(phase):
 import importlib.metadata as m
 write(RUN/(phase+'-source.json'),dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),corpus_manifest_sha256=sha(CORPUS/'manifest.json'),packages={k:m.version(k) for k in ['torch','transformers','tokenizers','numpy','safetensors']}))
def initialize():
 import torch
 from tokenizers import Tokenizer,models,pre_tokenizers,decoders,processors,trainers
 from transformers import BertConfig,BertModel,AutoTokenizer,PreTrainedTokenizerFast
 RUN.mkdir(parents=True,exist_ok=False);freeze('initialize');torch.manual_seed(17);torch.set_num_threads(6)
 data=rows();train=[r['text'] for r in data if r['split']=='train'];backend=Tokenizer(models.BPE(unk_token='[UNK]'));backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);backend.decoder=decoders.ByteLevel()
 trainer=trainers.BpeTrainer(vocab_size=32768,min_frequency=2,special_tokens=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
 backend.train_from_iterator(train,trainer=trainer);backend.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',pair='[CLS] $A [SEP] $B:1 [SEP]:1',special_tokens=[('[CLS]',2),('[SEP]',3)])
 tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]',model_max_length=256);tok.save_pretrained(RUN/'tokenizer')
 for s in ['验证码🙂👩🏽‍💻','é e\u0301\t中文\nEnglish','العربية हिन्दी','\x00<mask>\u200b']:
  ids=tok.encode(s,add_special_tokens=False);assert 1 not in ids and tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==s
 pre=MODELS/'lealla';meta=json.loads((pre/'source-manifest.json').read_text())
 for name,record in meta['files'].items():assert sha(pre/name)==record['sha256']
 old=BertModel.from_pretrained(pre,local_files_only=True,use_safetensors=True,attn_implementation='eager');oldtok=AutoTokenizer.from_pretrained(pre,local_files_only=True)
 cfg=BertConfig(**{**old.config.to_dict(),'vocab_size':len(tok),'num_hidden_layers':6,'pad_token_id':0,'use_cache':False});cfg._attn_implementation='eager';model=BertModel(cfg)
 layers=[3,7,11,15,19,23]
 for i,j in enumerate(layers):model.encoder.layer[i].load_state_dict(old.encoder.layer[j].state_dict())
 for name in ['position_embeddings','token_type_embeddings','LayerNorm']:getattr(model.embeddings,name).load_state_dict(getattr(old.embeddings,name).state_dict())
 model.pooler.load_state_dict(old.pooler.state_dict())
 mapped=0;random=0;special=tok.all_special_tokens;oldweights=old.embeddings.word_embeddings.weight
 with torch.no_grad():
  for i in range(len(tok)):
   piece=tok.convert_ids_to_tokens(i)
   if piece in special:ids=[oldtok.convert_tokens_to_ids(piece)]
   else:
    text=backend.decoder.decode([piece]);ids=oldtok.encode(text,add_special_tokens=False) if '\ufffd' not in text else []
   if ids and oldtok.unk_token_id not in ids:model.embeddings.word_embeddings.weight[i].copy_(oldweights[ids].mean(0));mapped+=1
   else:random+=1
  model.embeddings.word_embeddings.weight[0].zero_()
 model.save_pretrained(RUN/'initialized',safe_serialization=True);tok.save_pretrained(RUN/'initialized');params=sum(p.numel() for p in model.parameters());del old;gc.collect()
 teacher_tok=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True);n=len(data);tokens=np.lib.format.open_memmap(RUN/'tokens.npy',mode='w+',dtype=np.uint16,shape=(n,256));lengths=np.zeros(n,np.uint16);teacherlengths=np.zeros(n,np.uint16);visible=[];clipped=collections.Counter()
 for i,r in enumerate(data):
  text=r['text'];enc=tok(text,return_offsets_mapping=True,add_special_tokens=True,truncation=False);ids=enc['input_ids'];teacherids=teacher_tok.encode(text,add_special_tokens=True)
  if len(ids)>256 or len(teacherids)>256:
   if len(ids)>256:text=text[:enc['offset_mapping'][254][1]]
   while len(tok.encode(text,add_special_tokens=True))>256 or len(teacher_tok.encode(text,add_special_tokens=True))>256:text=text[:-1]
   ids=tok.encode(text,add_special_tokens=True);teacherids=teacher_tok.encode(text,add_special_tokens=True);clipped[r['split']]+=1
  assert ids and 1 not in ids and len(ids)<=256 and len(teacherids)<=256
  tokens[i,:]=0;tokens[i,:len(ids)]=ids;lengths[i]=len(ids);teacherlengths[i]=len(teacherids);visible.append(dict(id=r['id'],text=text,split=r['split'],family=r['family'],language=r['language'],clipped=text!=r['text']))
  if i%20000==0:print('ALIGNED',i,n,flush=True)
 tokens.flush();np.save(RUN/'lengths.npy',lengths);np.save(RUN/'teacher-lengths.npy',teacherlengths)
 with (RUN/'visible.jsonl').open('x',encoding='utf8') as f:
  for r in visible:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 pairs=[json.loads(l) for l in (CORPUS/'pairs.jsonl').read_text().splitlines()];index={r['id']:i for i,r in enumerate(data)};pairs=[(index[p['a']],index[p['b']]) for p in pairs if p['split']=='train' and not visible[index[p['a']]]['clipped'] and not visible[index[p['b']]]['clipped']];np.save(RUN/'train-pairs.npy',np.asarray(pairs,np.int64))
 write(RUN/'initialization.json',dict(parameters=params,tokenizer_vocab=len(tok),byte_level_full_alphabet=True,seed=17,layers=layers,word_embedding_mapped=mapped,word_embedding_random=random,teacher_student_same_visible_prefix=True,truncated=dict(clipped),train_rows=len(train),rows=n,pairs=len(pairs),weights_sha256=sha(RUN/'initialized/model.safetensors'),tokenizer_sha256=sha(RUN/'tokenizer/tokenizer.json'),tokens_sha256=sha(RUN/'tokens.npy'),visible_sha256=sha(RUN/'visible.jsonl'),data_revision_sha256=sha(CORPUS/'manifest.json')));print('INITIALIZED',params,mapped,random,flush=True)
def teacher():
 import torch
 from transformers import AutoTokenizer,AutoModel
 freeze('teacher');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert sha(TEACHER/'pytorch_model.bin')=='b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38'
 data=[json.loads(l) for l in (RUN/'visible.jsonl').read_text().splitlines()];init=json.loads((RUN/'initialization.json').read_text());assert sha(RUN/'visible.jsonl')==init['visible_sha256']
 tok=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True);model=AutoModel.from_pretrained(TEACHER,local_files_only=True,use_safetensors=False,weights_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().to('cuda');out=np.lib.format.open_memmap(RUN/'teacher-fp32.npy',mode='w+',dtype=np.float32,shape=(len(data),1024));lengths=np.load(RUN/'teacher-lengths.npy');order=np.argsort(lengths,kind='stable');started=time.perf_counter();done=0
 with torch.inference_mode():
  for pos in range(0,len(order),32):
   idx=order[pos:pos+32];batch=tok([data[i]['text'] for i in idx],padding=True,truncation=False,return_tensors='pt');assert batch['input_ids'].shape[1]<=256
   h=model(**{k:v.to('cuda') for k,v in batch.items()}).last_hidden_state[:,0].float();v=torch.nn.functional.normalize(h,dim=1).cpu().numpy();assert np.isfinite(v).all();out[idx]=v;done+=len(idx)
   if pos%1024==0:out.flush();print(json.dumps(dict(rows=done,total=len(data),seconds=time.perf_counter()-started,rows_per_second=done/(time.perf_counter()-started))),flush=True)
 out.flush();np.testing.assert_allclose(np.linalg.norm(out,axis=1),1,atol=1e-5)
 write(RUN/'teacher-complete.json',dict(rows=len(data),dim=1024,seconds=time.perf_counter()-started,precision='FP32 weights and eager forward; TF32 disabled',max_cuda_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(0),weights_sha256=sha(TEACHER/'pytorch_model.bin'),features_sha256=sha(RUN/'teacher-fp32.npy'),visible_sha256=sha(RUN/'visible.jsonl')));print('TEACHER_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','teacher']);a=p.parse_args();{'initialize':initialize,'teacher':teacher}[a.phase]()
