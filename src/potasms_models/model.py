"""Versioned data-only packages and one inference/personalization interface."""
from pathlib import Path
import base64,ctypes as C,hashlib,json,math
import numpy as np
from . import native
from .data import digest
from .encoder import Encoder
from .evaluation import apply_scores

def fingerprint(manifest):
 data={k:manifest[k] for k in ['engine','featureContract','dimensions','files','scoring']}
 return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Model:
 """Engine selection changes representations, not the consumer's predict API."""
 def __init__(self,directory,encoder_library=None):
  self.closed=False;self.directory=Path(directory).resolve();path=self.directory/'model.json'
  if path.stat().st_size>65536:raise ValueError('Oversized manifest')
  m=json.loads(path.read_text());self.manifest=m
  if m.get('formatVersion')!=1 or m.get('engine') not in ['character-linear-int8-v1','compact-bert-q8-v1']:raise ValueError('Unsupported model engine or format')
  self.identity=fingerprint(m);self.dim=int(m['dimensions']);self.threshold=float(m['scoring']['thresholdLogit']);self.encoder=None
  if m.get('modelFingerprint',self.identity)!=self.identity:raise ValueError('Model fingerprint mismatch')
  if not math.isfinite(self.threshold) or not 0<self.dim<=1048576:raise ValueError('Invalid model dimensions or threshold')
  required={'base.v1.json'} if m['engine']=='character-linear-int8-v1' else {'encoder.q8_0.gguf','encoder.q8_0.gguf.pooler.bin','head.f32.bin'}
  if set(m['files'])!=required or not -120<=self.threshold<=120:raise ValueError('Unsupported model payload or threshold')
  scoring=m['scoring'];cal=scoring.get('calibration')
  if m['engine']=='compact-bert-q8-v1':
   if not isinstance(cal,dict) or not {'center','scale','log_slope','intercept'}.issubset(cal):raise ValueError('Invalid calibration contract')
   if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in [cal[k] for k in ['center','scale','log_slope','intercept']]) or abs(cal['center'])>10000 or not 0<cal['scale']<=10000 or not -4<=cal['log_slope']<=4 or not -10<=cal['intercept']<=10 or not math.isfinite(scoring['bias']) or abs(scoring['bias'])>1000:raise ValueError('Invalid calibration values')
  elif cal is not None:raise ValueError('Character package uses its original probability policy')

  for name,item in m['files'].items():
   file=self.directory/name
   if name in ['.','..'] or Path(name).name!=name or file.is_symlink() or not file.is_file() or file.stat().st_size!=item['bytes'] or item['bytes']>64*1024**2 or digest(file)!=item['sha256']:raise ValueError('Invalid model payload: '+name)
  if m['engine']=='character-linear-int8-v1':
   self.base=json.loads((self.directory/'base.v1.json').read_text());self.weights=np.frombuffer(base64.b64decode(self.base['quantizedWeights'],validate=True),np.int8).copy()
   if self.base['featureContract']!=m['featureContract'] or m['featureContract'] not in [native.LEGACY_CONTRACT,native.CONTRACT] or len(self.weights)!=self.dim or self.dim!=self.base['buckets']:raise ValueError('Character feature contract mismatch')
   if not math.isfinite(self.base['bias']) or abs(self.base['bias'])>1000 or not 0<self.base['weightScale']<=100 or not 0<self.base['threshold']<1 or abs(self.threshold-math.log(self.base['threshold']/(1-self.base['threshold'])))>1e-6:raise ValueError('Invalid character scoring values')
  else:
   if self.dim!=128 or m['featureContract']!='compact-bert-bpe32768-pooler128-v1' or encoder_library is None:raise ValueError('Missing supported native encoder')
   self.weights=np.fromfile(self.directory/'head.f32.bin',dtype='<f4')
   if self.weights.shape!=(128,) or not np.isfinite(self.weights).all() or np.max(abs(self.weights))>1000:raise ValueError('Invalid float head')
   self.encoder=Encoder(encoder_library,self.directory/'encoder.q8_0.gguf')
  self.lib=native.library();self.lib.pota_float_logit.argtypes=[C.POINTER(native.Feature),C.c_size_t,C.POINTER(C.c_float),C.c_size_t,C.c_float,C.POINTER(C.c_float),C.c_float,C.POINTER(C.c_float)];self.lib.pota_pa2_margin_sample.argtypes=[C.POINTER(native.Feature),C.c_size_t,C.POINTER(C.c_float),C.c_size_t,C.c_float,C.c_uint8,C.c_float,C.c_float,C.c_float];self.lib.pota_pa2_sample.argtypes=[C.POINTER(native.Feature),C.c_size_t,C.POINTER(C.c_int8),C.c_size_t,C.c_float,C.c_float,C.POINTER(C.c_float),C.c_float,C.c_float,C.c_uint8,C.c_float,C.c_float,C.c_float]
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def close(self):
  self.closed=True
  if self.encoder:self.encoder.close();self.encoder=None
 def _features(self,text):
  if self.closed:raise ValueError('Model is closed')
  if len(text.encode())>16384:raise ValueError('SMS exceeds bounds')
  if self.encoder:items=list(enumerate(self.encoder.encode(text)))
  else:items=native.features(text,buckets=self.dim,contract=self.manifest['featureContract'])
  return (native.Feature*len(items))(*(native.Feature(i,v) for i,v in items))
 def _raw(self,features):
  out=C.c_float()
  if self.manifest['engine']=='character-linear-int8-v1':
   status=self.lib.pota_logit(features,len(features),self.weights.ctypes.data_as(C.POINTER(C.c_int8)),self.dim,self.base['weightScale'],self.base['bias'],None,0,C.byref(out))
  else:status=self.lib.pota_float_logit(features,len(features),self.weights.ctypes.data_as(C.POINTER(C.c_float)),self.dim,self.manifest['scoring']['bias'],None,0,C.byref(out))
  native.check(status);raw=float(out.value);c=self.manifest['scoring'].get('calibration');return float(apply_scores(raw,c)) if c else raw
 def empty(self):return {'parent':self.identity,'delta':np.zeros(self.dim,np.float32)}
 def _validate(self,state):
  if state['parent']!=self.identity:raise ValueError('Personal weights belong to another model version')
  delta=state['delta']
  if delta.shape!=(self.dim,) or delta.dtype!=np.float32 or not np.isfinite(delta).all() or np.max(abs(delta))>2 or np.linalg.norm(delta.astype(float))>12.0001:raise ValueError('Invalid personal state')
 def predict(self,text,state=None):
  state=self.empty() if state is None else state;self._validate(state);f=self._features(text);base=np.float32(self._raw(f)-self.threshold);change=np.float32(0)
  for v in f:change=np.float32(change+np.float32(v.value*state['delta'][v.bucket]))
  margin=float(np.float32(base+change));return {'junk':margin>=0,'decisionMargin':margin,'probability':float(self.lib.pota_sigmoid(margin+self.threshold)),'modelFingerprint':self.identity}
 def learn(self,examples,state=None,cancelled=lambda:False):
  state=self.empty() if state is None else state;self._validate(state);delta=state['delta'].copy();prepared=[]
  for text,label in examples:
   if label not in (0,1) or not text.strip():raise ValueError('Invalid confirmed feedback')
   f=self._features(text);prepared.append((f,label,np.float32(self._raw(f)-self.threshold)))
  for epoch in range(24):
   for f,label,margin in prepared:
    if cancelled():raise InterruptedError('Learning cancelled; input state unchanged')
    ptr=delta.ctypes.data_as(C.POINTER(C.c_float))
    if self.encoder:status=self.lib.pota_pa2_margin_sample(f,len(f),ptr,self.dim,margin,label,1.,.05,2.)
    else:status=self.lib.pota_pa2_sample(f,len(f),self.weights.ctypes.data_as(C.POINTER(C.c_int8)),self.dim,self.base['weightScale'],self.base['bias'],ptr,0.,self.threshold,label,1.,.05,2.)
    native.check(status)
   norm=np.linalg.norm(delta.astype(float))
   if norm>12:delta=(delta.astype(float)*12/norm).astype(np.float32)
  delta[abs(delta)<np.float32(.0001)]=0;return {'parent':self.identity,'delta':delta}
