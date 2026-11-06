from pathlib import Path
import json,os,subprocess,tarfile
from .data import ROOT,digest,fetch

def build_native():
 out=ROOT/'build/native'
 subprocess.run(['cmake','-S',str(ROOT/'native'),'-B',str(out),'-DCMAKE_BUILD_TYPE=Release'],check=True)
 subprocess.run(['cmake','--build',str(out),'--parallel','4'],check=True)
 candidates=[out/'libpotasms.so',out/'libpotasms.dylib',out/'Release/potasms.dll']
 lib=next((p for p in candidates if p.exists()),None)
 if lib is None:raise RuntimeError('Native library not produced')
 os.environ['POTASMS_NATIVE_LIB']=str(lib)
 return lib

def semantic(work,archive=None):
 work=Path(work);spec=json.loads((ROOT/'manifests/semantic-v1.json').read_text());archive=Path(archive) if archive else fetch(spec['url'],work/'downloads'/spec['asset'],spec['sha256'])
 if digest(archive)!=spec['sha256']:raise ValueError('Semantic snapshot hash mismatch')
 target=work/'corpora/processed';target.mkdir(parents=True,exist_ok=False)
 expected=spec['manifest']['files'];seen=set();total=0
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   parts=Path(member.name).parts
   if len(parts)!=2 or parts[0]!='semantic-v1' or not member.isfile() or member.issym() or member.islnk():raise ValueError('Unexpected semantic archive entry')
   name=parts[1]
   if name in seen or name not in {*expected,'manifest.json'}:raise ValueError('Duplicate or unlisted semantic entry')
   total+=member.size
   if member.size>256*1024**2 or total>512*1024**2:raise ValueError('Semantic archive exceeds bounds')
   seen.add(name)
   with tar.extractfile(member) as src,(target/name).open('xb') as dst:
    while block:=src.read(1024*1024):dst.write(block)
 if seen!={*expected,'manifest.json'}:raise ValueError('Semantic archive file inventory mismatch')
 for name,item in expected.items():
  if digest(target/name)!=item['sha256']:raise ValueError('Semantic member hash mismatch: '+name)
 if json.loads((target/'manifest.json').read_text())!=spec['manifest']:raise ValueError('Semantic manifest differs')
 return target

def models(work):
 from huggingface_hub import hf_hub_download
 work=Path(work);pins=json.loads((ROOT/'manifests/pretrained-models.json').read_text())
 for name,spec in pins.items():
  target=work/'models'/name;target.mkdir(parents=True,exist_ok=True)
  for filename,record in spec['files'].items():
   path=Path(hf_hub_download(spec['repo'],filename,revision=spec['revision'],local_dir=str(target),cache_dir=str(work/'cache/huggingface'),token=False))
   if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:raise ValueError('Pretrained source hash mismatch')
  (target/'source-manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
 return work/'models'
