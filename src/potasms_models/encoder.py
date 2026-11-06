"""ctypes transport only; inference is the native C++/Rust CPU runtime."""
import ctypes as C,os
import numpy as np

def logits(x,weights,bias):
 from . import native
 lib=native.library();lib.pota_float_logit.argtypes=[C.POINTER(native.Feature),C.c_size_t,C.POINTER(C.c_float),C.c_size_t,C.c_float,C.POINTER(C.c_float),C.c_float,C.POINTER(C.c_float)];w=np.ascontiguousarray(weights,np.float32);out=[]
 for row in x:
  f=(native.Feature*len(row))(*(native.Feature(i,v) for i,v in enumerate(row)));value=C.c_float();native.check(lib.pota_float_logit(f,len(row),w.ctypes.data_as(C.POINTER(C.c_float)),len(w),float(bias),None,0.,C.byref(value)));out.append(value.value)
 return np.asarray(out,dtype=float)

class Encoder:
 def __init__(self,library,model):
  os.environ['EMBEDDINGS_CPP_BACKEND']='cpu';os.environ['EMBEDDINGS_CPP_THREADS']='4';os.environ['EMBEDDINGS_CPP_CPU_REPACK']='0';os.environ['EMBEDDINGS_CPP_BERT_FLASH_ATTN']='0'
  self.lib=C.CDLL(str(library));self.lib.pota_semantic_open.argtypes=[C.c_char_p,C.c_int,C.c_int];self.lib.pota_semantic_open.restype=C.c_void_p;self.lib.pota_semantic_encode.argtypes=[C.c_void_p,C.c_char_p,C.c_size_t,C.POINTER(C.c_float),C.c_size_t,C.POINTER(C.c_size_t)];self.lib.pota_semantic_close.argtypes=[C.c_void_p];self.lib.pota_semantic_error.restype=C.c_char_p;self.handle=self.lib.pota_semantic_open(os.fsencode(model),256,1)
  if not self.handle:raise ValueError(self.lib.pota_semantic_error().decode())
 def encode(self,text):
  raw=text.encode('utf8')
  if len(raw)>16384:raise ValueError('SMS exceeds16384UTF8bytes')
  v=np.zeros(128,np.float32);n=C.c_size_t();rc=self.lib.pota_semantic_encode(self.handle,raw,len(raw),v.ctypes.data_as(C.POINTER(C.c_float)),len(v),C.byref(n))
  if rc or n.value!=128:raise ValueError(self.lib.pota_semantic_error().decode())
  if not np.isfinite(v).all() or abs(float(np.linalg.norm(v))-1)>1e-5:raise ValueError('Invalid native embedding')
  return v
 def close(self):
  if self.handle:self.lib.pota_semantic_close(self.handle);self.handle=None
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
