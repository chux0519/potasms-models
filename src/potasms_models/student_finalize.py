from pathlib import Path
import hashlib,json,platform,shutil
import numpy as np
from .data import digest
from .evaluation import calibrate_scores,apply_scores,common_language_threshold,panels
from .encoder import Encoder

def finalize(work):
 from .paths import RUN
 from .student_export import export
 work=Path(work);out=work/'student-export';export(RUN/'c24-sms-fit/selection.json',out)
 runtime=json.loads((work/'runtime-build/manifest.json').read_text());library=work/runtime['encoder_library'];rows=[json.loads(l) for l in (work/'sms-v1/rows.jsonl').read_text().splitlines()];y=np.array([r['label'] for r in rows]);cal=np.flatnonzero([r['pool']=='calibration' for r in rows]);languages=[rows[i]['language'] for i in cal];head=np.load(RUN/'evaluation/heads/c24-supervised.npz');results={}
 for precision in ['f32','f16','q8_0']:
  with Encoder(library,out/f'encoder.{precision}.gguf') as model:x=np.stack([model.encode(r['text']) for r in rows])
  np.savez(out/(precision+'-sms.npz'),x=x);raw=x@head['w']+float(head['bias']);calibration=calibrate_scores(y[cal],raw[cal]);z=apply_scores(raw,calibration);point=common_language_threshold(y[cal],z[cal],languages,.001);results[precision]={'panels':panels(rows,z,point['threshold']),'calibration':calibration,'point':point,'encoder_sha256':digest(out/f'encoder.{precision}.gguf'),'feature_bytes_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'float_head_refit':False,'backend':'nativeCPU4/repack0/standardattention','runtime_sha256':runtime['encoder_sha256']}
  (out/(precision+'-calibration.json')).write_text(json.dumps({'calibration':calibration,'point':point},indent=2)+'\n');print('NATIVE_STUDENT_EVALUATED',precision,results[precision]['panels']['background'],flush=True)
 shutil.copy2(RUN/'evaluation/heads/c24-supervised.npz',out/'head.npz');(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');return results
