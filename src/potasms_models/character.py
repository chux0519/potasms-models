"""From-scratch legacy character recipe on the exact shared SMS label budget."""
from pathlib import Path
import base64,ctypes as C,hashlib,importlib.metadata,json,time
import numpy as np
from scipy.sparse import csr_matrix,save_npz
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from . import native
from .data import digest
from .evaluation import calibrate_scores,apply_scores,common_language_threshold,choose_threshold,panels

def matrix(rows,contract=native.LEGACY_CONTRACT):
 indptr=[0];indices=[];values=[]
 for row in rows:
  for index,value in native.features(row['text'],contract=contract):indices.append(index);values.append(value)
  indptr.append(len(indices))
 return csr_matrix((np.asarray(values,np.float32),np.asarray(indices,np.int32),np.asarray(indptr,np.int64)),shape=(len(rows),65536))

def logits(base,rows):
 w=np.frombuffer(base64.b64decode(base['quantizedWeights']),np.int8).copy();out=[];lib=native.library()
 for row in rows:
  fs=native.features(row['text'],buckets=len(w),contract=base['featureContract']);features=(native.Feature*len(fs))(*(native.Feature(*r) for r in fs));delta=(C.c_float*len(fs))();v=C.c_float()
  native.check(lib.pota_logit(features,len(fs),w.ctypes.data_as(C.POINTER(C.c_int8)),len(w),base['weightScale'],base['bias'],delta,0.,C.byref(v)));out.append(v.value)
 return np.asarray(out,dtype=float)

def train(work):
 work=Path(work);out=work/'character';out.mkdir(parents=True,exist_ok=False)
 source=work/'sms-v1/rows.jsonl';rows=[json.loads(l) for l in source.read_text().splitlines()];assert len(rows)==5261
 x=matrix(rows);save_npz(out/'features-v1.npz',x);y=np.array([r['label'] for r in rows]);tr=np.flatnonzero([r['pool']=='train' for r in rows]);cal=np.flatnonzero([r['pool']=='calibration' for r in rows]);cl=[rows[i]['language'] for i in cal];assert len(tr)==1024 and len(cal)==768;counts=np.bincount(y[tr],minlength=2);weight=np.array([len(tr)/(2*counts[v]) for v in y[tr]])
 protocol=dict(recipe='SGD log_loss,alpha1e-6,max_iter200,tol1e-3,balanced sample weights,INT8 symmetric',seeds=[42,17,73],primary_seed=42,feature_contract=native.LEGACY_CONTRACT,sms_sha256=digest(source),train_rows=1024,calibration_rows=768,old_weights_loaded=False,private_data=False,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn']});(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');report={}
 for seed in protocol['seeds']:
  folder=out/f'seed-{seed}';folder.mkdir();start=time.perf_counter();fit=SGDClassifier(loss='log_loss',alpha=1e-6,max_iter=200,tol=1e-3,random_state=seed).fit(x[tr],y[tr],sample_weight=weight);w=fit.coef_.ravel().astype(np.float32);scale=float(abs(w).max())/127 or 1.;q=np.clip(np.rint(w/scale),-127,127).astype(np.int8);bias=float(fit.intercept_[0]);threshold=max(choose_threshold(y[cal],expit((x@w+bias)[cal]),.001),.995)
  if not 0<threshold<1:raise ValueError('Legacy probability threshold cannot represent reject-all; retain failure rather than silently changing policy')
  ident=hashlib.sha256(q.tobytes()+np.asarray([scale,bias,threshold],np.float32).tobytes()).hexdigest()[:12]
  base=dict(formatVersion=1,modelVersion='potasms-public-char-'+ident,featureContract=native.LEGACY_CONTRACT,weightScale=scale,bias=bias,threshold=threshold,buckets=65536,minN=2,maxN=5,quantizedWeights=base64.b64encode(q.tobytes()).decode(),attribution=['Original authored synthetic SMS scenarios (2026-11-05)','UCI SMS Spam Collection, DOI10.24432/C5CC84, CC BY4.0','Agarwal et al., IMC2025 Fishing for Smishing, CC BY4.0']);path=folder/'base.v1.json';path.write_text(json.dumps(base,separators=(',',':'))+'\n');z=logits(base,rows);error=float(abs(z-(x@(q.astype(np.float32)*np.float32(scale))+np.float32(bias))).max());assert error<2e-4,error
  t=float(np.log(threshold/(1-threshold)));prob=np.asarray([native.library().pota_sigmoid(float(v)) for v in z]);np.testing.assert_array_equal(z>=t,prob>=threshold)
  calibration=calibrate_scores(y[cal],z[cal]);score=apply_scores(z,calibration);matched=common_language_threshold(y[cal],score[cal],cl,.001)
  report[str(seed)]={'legacy':panels(rows,z,t),'matched':panels(rows,score,matched['threshold']),'model_sha256':digest(path),'model_version':base['modelVersion'],'iterations':int(fit.n_iter_),'seconds':time.perf_counter()-start,'native_max_logit_difference':error}
  (folder/'matched-calibration.json').write_text(json.dumps({'calibration':calibration,'point':matched},indent=2)+'\n');np.savez(folder/'scores.npz',raw_logit=z,native_probability=prob,matched_logit=score)
  print('CHARACTER_TRAINED',seed,report[str(seed)]['legacy']['background'],flush=True)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');return report
