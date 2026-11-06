"""Bounded label-supervised correction with semantic replay; train-only hard mining."""
from pathlib import Path
import json,math,os,subprocess,time,argparse
import numpy as np
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from .student_prepare_model import RUN,sha,write
from .student_features import OUT,PARENT

def fit(selection_file=None,out_name="sms-fit-01",head_name="s5-distilled",checkpointing=False):
 import torch
 import torch.nn.functional as F
 from transformers import AutoModel,AutoTokenizer
 from safetensors.torch import load_file,save_file
 out=RUN/out_name;out.mkdir(exist_ok=False);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 selected=json.loads((selection_file or RUN/'fit-01/selection.json').read_text());checkpoint=Path(selected['checkpoint']);assert sha(checkpoint/'model.safetensors')==selected['weights_sha256'];records=[json.loads(l) for l in (OUT/'sms-rows.jsonl').read_text().splitlines()];tr=np.array([i for i,r in enumerate(records) if r['pool']=='train']);dev=np.array([i for i,r in enumerate(records) if r['pool']=='dev']);y=np.array([r['label'] for r in records]);assert len(tr)==1024 and len(dev)==192
 assert {records[i]['source'] for i in tr}<={'original-authored-20261105','uci_sms_2012','reportsmishing_imc25'}
 tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True);sms=tok([r['text'] for r in records],padding=True,truncation=True,max_length=256,return_tensors='np');lengths=sms['attention_mask'].sum(1);teacher_sms=np.load(PARENT/'encodings/bge-fp32.npz')['x'];sim=teacher_sms[tr]@teacher_sms[tr].T;sim[y[tr,None]==y[tr][None,:]]=-2;negative=tr[np.argmax(sim,axis=1)];assert np.all(y[negative]!=y[tr])
 generic=[json.loads(l) for l in (RUN/'visible.jsonl').read_text().splitlines()];gt=np.array([i for i,r in enumerate(generic) if r['split']=='train']);gd=np.array([i for i,r in enumerate(generic) if r['split']=='dev'])[:2048];tokens=np.load(RUN/'tokens.npy',mmap_mode='r');gl=np.load(RUN/'lengths.npy');teacher=np.load(RUN/'teacher-fp32.npy',mmap_mode='r')
 model=AutoModel.from_pretrained(checkpoint,local_files_only=True,use_safetensors=True,attn_implementation='eager').to('cuda');projection=torch.nn.Linear(128,1024,bias=False).to('cuda');projection.weight.data.copy_(load_file(str(checkpoint/'training-projection.safetensors'))['weight'].to('cuda'));projection.requires_grad_(False);head=torch.nn.Linear(128,2).to('cuda');
 if checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 old=np.load(OUT/'heads'/(head_name+'.npz'))
 with torch.no_grad():
  w=torch.tensor(old['w'],device='cuda');b=float(old['bias']);head.weight[0].copy_(-w/2);head.weight[1].copy_(w/2);head.bias[0]=-b/2;head.bias[1]=b/2
 optimizer=torch.optim.AdamW([*model.parameters(),*head.parameters()],lr=1e-4,weight_decay=.01);total=10*math.ceil(len(tr)/64);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min((n+1)/16,max((total-n)/(total-16),0)))
 def enc(ids,length):
  width=int(length.max());ids=torch.tensor(np.array(ids[:,:width],dtype=np.int64),device='cuda');return F.normalize(model(input_ids=ids,attention_mask=(ids!=0).long()).pooler_output.float(),dim=1)
 def validation(epoch):
  model.eval();head.eval();zz=[];sem=[]
  with torch.inference_mode():
   for start in range(0,len(dev),128):idx=dev[start:start+128];zz.append(head(enc(sms['input_ids'][idx],lengths[idx])).cpu())
   for start in range(0,len(gd),128):idx=gd[start:start+128];z=enc(tokens[idx],gl[idx]);p=F.normalize(projection(z),dim=1);sem.extend((p*torch.tensor(np.array(teacher[idx]),device='cuda')).sum(1).cpu().tolist())
  ce=float(F.cross_entropy(torch.cat(zz),torch.tensor(y[dev],dtype=torch.long)));cos=float(np.mean(sem));return dict(epoch=epoch,dev_ce=ce,semantic_dev_cosine=cos,selection_loss=ce+.1*(1-cos))
 protocol=dict(gradient_checkpointing=checkpointing,epochs=10,steps=total,batch=64,seed=17,lr=1e-4,hard_negative_cosine_margin=.4,loss_weights={'sms_ce':1.,'hard_negative_squared_hinge':.5,'semantic_teacher_cosine':.25,'semantic_relation':.2},negative_mining='BGE cosine nearest opposite-label TRAIN row only',sources=sorted({records[i]['source'] for i in tr}),source_counts={s:sum(records[i]['source']==s for i in tr) for s in sorted({records[i]['source'] for i in tr})},selection='minimum devCE + .1*(1-semantic-dev-cosine); test/calibration never optimized',parent_checkpoint_sha256=selected['weights_sha256'],sms_input_sha256=sha(OUT/'sms-rows.jsonl'),source_sha256=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),research_only=True,reason='first exposed SMS regression shows semantic-only student collateral FP; this is a followup, not fresh independent validation');write(out/'protocol.json',protocol);np.save(out/'hard-negative-train-indices.npy',np.c_[tr,negative]);best=None;step=0;rng=np.random.default_rng(17);started=time.perf_counter();write(out/'dev-0.json',validation(0))
 for epoch in range(1,11):
  model.train();head.train();order=rng.permutation(len(tr));sums=np.zeros(4)
  for pos in range(0,len(order),64):
   ids=order[pos:pos+64];actual=tr[ids];opposite=negative[ids];combined=np.r_[actual,opposite];optimizer.zero_grad(set_to_none=True);z=enc(sms['input_ids'][combined],lengths[combined]);ce=F.cross_entropy(head(z),torch.tensor(y[combined],device='cuda'));hard=F.relu((z[:len(ids)]*z[len(ids):]).sum(1)-.4).square().mean();idx=rng.choice(gt,64,replace=False);s=enc(tokens[idx],gl[idx]);p=F.normalize(projection(s),dim=1);t=torch.tensor(np.array(teacher[idx]),device='cuda');kd=(1-(p*t).sum(1)).mean();rel=(s@s.T-t@t.T).square().mean();loss=ce+.5*hard+.25*kd+.2*rel;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*head.parameters()],1.);optimizer.step();scheduler.step();step+=1;sums+=np.array([float(ce.detach()),float(hard.detach()),float(kd.detach()),float(rel.detach())])
  result=validation(epoch);target=out/f'epoch-{epoch}';model.save_pretrained(target,safe_serialization=True);tok.save_pretrained(target);save_file({'weight':head.weight.detach().cpu().contiguous(),'bias':head.bias.detach().cpu().contiguous()},str(target/'sms-head.safetensors'));save_file({'weight':projection.weight.detach().cpu().contiguous()},str(target/'training-projection.safetensors'));result.update(checkpoint=str(target),weights_sha256=sha(target/'model.safetensors'),loss_components=(sums/math.ceil(len(tr)/64)).tolist());write(out/f'dev-{epoch}.json',result)
  if best is None or result['selection_loss']<best['selection_loss']:best=result;write(out/'selection.json',best)
  print('SMS_EPOCH',json.dumps(result),flush=True)
 write(out/'complete.json',dict(best=best,steps=step,seconds=time.perf_counter()-started,max_cuda_bytes=torch.cuda.max_memory_allocated(),research_only=True));print('SMS_FIT_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--selection',type=Path);p.add_argument('--out-name',default='sms-fit-01');p.add_argument('--head-name',default='s5-distilled');p.add_argument('--checkpointing',action='store_true');a=p.parse_args();fit(a.selection,a.out_name,a.head_name,a.checkpointing)
