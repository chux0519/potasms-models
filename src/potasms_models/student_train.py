"""S5 semantic distillation; selection uses semantic dev only, never SMS test."""
from pathlib import Path
import hashlib,json,math,os,subprocess,time,argparse
import numpy as np
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from .student_prepare_model import RUN,sha,write

def train(initial_path=None,fit_name="fit-01"):
 import torch
 import torch.nn.functional as F
 from transformers import BertModel
 from safetensors.torch import save_file
 initial_path=initial_path or RUN/'initialized';out=RUN/fit_name;out.mkdir(exist_ok=False);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 meta=json.loads((RUN/'teacher-complete.json').read_text());assert sha(RUN/'teacher-fp32.npy')==meta['features_sha256'];records=[json.loads(l) for l in (RUN/'visible.jsonl').read_text().splitlines()];tokens=np.load(RUN/'tokens.npy',mmap_mode='r');lengths=np.load(RUN/'lengths.npy');teacher=np.load(RUN/'teacher-fp32.npy',mmap_mode='r');tr=np.array([i for i,r in enumerate(records) if r['split']=='train']);dev=np.array([i for i,r in enumerate(records) if r['split']=='dev'])[:2048]
 pairs=np.load(RUN/'train-pairs.npy');partners={}
 for a,b in pairs:partners.setdefault(int(a),int(b));partners.setdefault(int(b),int(a))
 model=BertModel.from_pretrained(initial_path,local_files_only=True,use_safetensors=True,attn_implementation='eager').to('cuda');projection=torch.nn.Linear(128,1024,bias=False).to('cuda');optimizer=torch.optim.AdamW([*model.parameters(),*projection.parameters()],lr=3e-4,weight_decay=.01);steps_per_epoch=math.ceil(len(tr)/64);total_steps=steps_per_epoch*3;warmup=total_steps//10
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warmup,max((total_steps-s)/(total_steps-warmup),0)))
 protocol=dict(seed=17,batch=64,epochs=3,steps_per_epoch=steps_per_epoch,total_steps=total_steps,lr=3e-4,warmup_steps=warmup,weight_decay=.01,gradient_clip=1.,weights_dtype='FP32',execution_dtype='FP32 eager, deterministic, TF32 off',objective={'projected_teacher_cosine':1.,'native_pair_relation_mse':.2,'projected_translation_partner_cosine':.1},selection='minimum 1-cosine to BGE on frozen first2048 hashed semantic-dev rows only',teacher_features_sha256=meta['features_sha256'],initial_weights_sha256=sha(initial_path/'model.safetensors'),source_sha256=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parameters=sum(p.numel() for p in model.parameters()),training_only_projection_parameters=sum(p.numel() for p in projection.parameters()))
 write(out/'protocol.json',protocol)
 def encode(idx):
  width=int(lengths[idx].max());ids=torch.tensor(np.array(tokens[idx,:width],dtype=np.int64),device='cuda');mask=(ids!=0).long();s=F.normalize(model(input_ids=ids,attention_mask=mask).pooler_output.float(),dim=1);return s,F.normalize(projection(s),dim=1)
 def validate(epoch):
  model.eval();projection.eval();ss=[];pp=[]
  with torch.inference_mode():
   for start in range(0,len(dev),128):
    s,p=encode(dev[start:start+128]);ss.append(s.cpu().numpy());pp.append(p.cpu().numpy())
  s=np.concatenate(ss);p=np.concatenate(pp);t=np.array(teacher[dev]);cos=(p*t).sum(1);native_rel_rmse=float(np.sqrt(np.mean((s[:512]@s[:512].T-t[:512]@t[:512].T)**2)))
  item=dict(epoch=epoch,rows=len(dev),teacher_cos_mean=float(cos.mean()),teacher_cos_min=float(cos.min()),native_relation_rmse=native_rel_rmse);write(out/f'dev-{epoch}.json',item);return item
 initial=validate(0);print('INITIAL_DEV',json.dumps(initial),flush=True);started=time.perf_counter();step=0;best=None
 for epoch in range(1,4):
  model.train();projection.train();order=np.random.default_rng(17+epoch).permutation(tr);sums=np.zeros(4);batches=0
  for pos in range(0,len(order),64):
   idx=order[pos:pos+64];optimizer.zero_grad(set_to_none=True);s,p=encode(idx);t=torch.tensor(np.array(teacher[idx]),device='cuda');kd=(1-(p*t).sum(1)).mean();relation=((s@s.T-t@t.T)**2).mean();available=[j for j,i in enumerate(idx) if int(i) in partners]
   if available:
    other=torch.tensor(np.array(teacher[[partners[int(idx[j])] for j in available]]),device='cuda');parallel=(1-(p[available]*other).sum(1)).mean()
   else:parallel=kd.new_zeros(())
   loss=kd+.2*relation+.1*parallel
   if not torch.isfinite(loss):raise ValueError('Nonfinite student loss')
   loss.backward();grad=float(torch.nn.utils.clip_grad_norm_([*model.parameters(),*projection.parameters()],1.));assert math.isfinite(grad);optimizer.step();scheduler.step();step+=1;batches+=1;sums+=np.array([float(loss.detach()),float(kd.detach()),float(relation.detach()),float(parallel.detach())])
   if step%100==0:print(json.dumps(dict(epoch=epoch,step=step,total=total_steps,loss=(sums/batches).tolist(),seconds=time.perf_counter()-started,max_cuda_bytes=torch.cuda.max_memory_allocated())),flush=True)
  metrics=validate(epoch);checkpoint=out/f'epoch-{epoch}';model.save_pretrained(checkpoint,safe_serialization=True)
  for file in (RUN/'tokenizer').glob('*'):
   if file.is_file():import shutil;shutil.copy2(file,checkpoint/file.name)
  save_file({'weight':projection.weight.detach().cpu().contiguous()},str(checkpoint/'training-projection.safetensors'));torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'step':step},checkpoint/'training-state.pt')
  result=dict(**metrics,checkpoint=str(checkpoint),weights_sha256=sha(checkpoint/'model.safetensors'),train_loss=(sums/batches).tolist())
  write(out/f'epoch-{epoch}.json',result)
  if best is None or result['teacher_cos_mean']>best['teacher_cos_mean']:best=result;write(out/'selection.json',best)
  print('EPOCH_DONE',json.dumps(result),flush=True)
 write(out/'complete.json',dict(best=best,steps=step,seconds=time.perf_counter()-started,max_cuda_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(0),selection_independent_of_sms=True,research_only=True));print('STUDENT_FIT_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--initial',type=Path);p.add_argument('--fit-name',default='fit-01');a=p.parse_args();train(a.initial,a.fit_name)
