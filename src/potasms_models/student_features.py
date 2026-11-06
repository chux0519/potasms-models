from .paths import RUN,MODELS,CORPUS
from .student_prepare_model import sha,write
import numpy as np,gc
OUT=RUN/'evaluation'
PARENT=RUN

def encode_context(path,texts):
 import torch
 from transformers import AutoModel,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModel.from_pretrained(path,local_files_only=True,use_safetensors=True,attn_implementation='eager').eval().to('cuda');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;parts=[]
 with torch.inference_mode():
  for i in range(0,len(texts),64):
   b=tok(texts[i:i+64],padding=True,truncation=True,max_length=256,return_tensors='pt');v=model(**{k:v.to('cuda') for k,v in b.items()}).pooler_output.float();parts.append(torch.nn.functional.normalize(v,dim=1).cpu().numpy())
 x=np.concatenate(parts);del model;gc.collect();torch.cuda.empty_cache();return x
