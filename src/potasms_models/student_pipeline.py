"""The published C24 recipe: initialize, distill, supervised correction, frozen head."""
from pathlib import Path
import gc,json,shutil,subprocess,sys
import numpy as np
from sklearn.linear_model import LogisticRegression

def train(work):
 from .paths import RUN,TEACHER
 from .student_prepare_model import initialize,teacher,sha,write
 from .student_train import train as distill
 from .student_sms_fit import fit
 from .student_features import OUT,encode_context
 import torch
 from transformers import AutoTokenizer,AutoModel
 work=Path(work)
 if RUN.exists():raise ValueError('Run directory exists; use a new work directory instead of overwriting')
 initialize()
 subprocess.run([sys.executable,'-m','potasms_models.student_deep_init'],check=True)
 teacher();distill(RUN/'c24-initialized','c24-fit')
 OUT.mkdir();shutil.copy2(work/'sms-v1/rows.jsonl',OUT/'sms-rows.jsonl');rows=[json.loads(l) for l in (OUT/'sms-rows.jsonl').read_text().splitlines()];texts=[r['text'] for r in rows];y=np.array([r['label'] for r in rows]);tr=np.flatnonzero([r['pool']=='train' for r in rows]);assert len(tr)==1024
 (RUN/'encodings').mkdir();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=AutoModel.from_pretrained(TEACHER,local_files_only=True,trust_remote_code=False,use_safetensors=False,weights_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().to('cuda');tok=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True,trust_remote_code=False);parts=[]
 # Same fixed256 padding / batch16 as the original frozen SMS teacher features.
 with torch.inference_mode():
  for pos in range(0,len(rows),16):
   ids=tok(texts[pos:pos+16],padding='max_length',truncation=True,max_length=256,return_tensors='pt');h=model(input_ids=ids['input_ids'].to('cuda'),attention_mask=ids['attention_mask'].to('cuda')).last_hidden_state.float()[:,0];parts.append(torch.nn.functional.normalize(h,p=2,dim=1).cpu().numpy())
 np.savez(RUN/'encodings/bge-fp32.npz',x=np.concatenate(parts));del model;gc.collect();torch.cuda.empty_cache()
 selected=json.loads((RUN/'c24-fit/selection.json').read_text());x=encode_context(Path(selected['checkpoint']),texts);np.savez(OUT/'c24-sms.npz',x=x);(OUT/'heads').mkdir();head=LogisticRegression(C=1.,max_iter=1000,solver='lbfgs').fit(x[tr],y[tr]);assert head.n_iter_.max()<1000;np.savez(OUT/'heads/c24.npz',w=head.coef_.ravel().astype(np.float32),bias=float(head.intercept_[0]))
 fit(RUN/'c24-fit/selection.json','c24-sms-fit','c24',checkpointing=True)
 selected=json.loads((RUN/'c24-sms-fit/selection.json').read_text());x=encode_context(Path(selected['checkpoint']),texts);np.savez(OUT/'c24-supervised-sms.npz',x=x);head=LogisticRegression(C=1.,max_iter=1000,solver='lbfgs').fit(x[tr],y[tr]);assert head.n_iter_.max()<1000;np.savez(OUT/'heads/c24-supervised.npz',w=head.coef_.ravel().astype(np.float32),bias=float(head.intercept_[0]))
 write(RUN/'public-training-complete.json',{'checkpoint_sha256':selected['weights_sha256'],'checkpoint':str(Path(selected['checkpoint']).relative_to(work)),'sms_rows_sha256':sha(work/'sms-v1/rows.jsonl'),'head_sha256':sha(OUT/'heads/c24-supervised.npz'),'private_data':False,'test_used_for_optimization':False,'research_only':True});print('PUBLIC_STUDENT_TRAINING_COMPLETE',flush=True)
