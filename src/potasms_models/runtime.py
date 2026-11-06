"""Build the pinned Python-free CPU encoder and quantizer from public sources."""
from pathlib import Path
import json,os,platform,subprocess,tarfile
from .data import ROOT,digest,fetch

def build(work,archive=None,offline=False):
 work=Path(work);spec=json.loads((ROOT/'manifests/runtime-v1.json').read_text());archive=Path(archive) if archive else fetch(spec['url'],work/'downloads'/spec['asset'],spec['sha256'])
 if digest(archive)!=spec['sha256']:raise ValueError('Runtime source archive hash mismatch')
 source=work/'runtime-source-v1';source.mkdir(parents=True,exist_ok=False);total=0
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   parts=Path(member.name).parts
   if len(parts)<2 or parts[0]!='runtime-source-v1' or '..' in parts or not member.isfile() or member.issym() or member.islnk():raise ValueError('Invalid runtime archive member')
   total+=member.size
   if total>256*1024**2:raise ValueError('Runtime source exceeds bounds')
   path=source.joinpath(*parts[1:]);path.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(member) as src,path.open('xb') as out:
    while block:=src.read(1024*1024):out.write(block)
 provenance=source/'PROVENANCE.json'
 if digest(provenance)!=spec['provenance_sha256']:raise ValueError('Runtime provenance mismatch')
 for name,record in json.loads(provenance.read_text())['files'].items():
  if digest(source/name)!=record['sha256']:raise ValueError('Runtime source member mismatch: '+name)
 build=work/'runtime-build';build.mkdir();env=dict(os.environ,CARGO_TARGET_DIR=str(build/'cargo'))
 subprocess.run(['cargo','build','--release','--locked',*(['--offline'] if offline else [])],cwd=source/'ffi-source',env=env,check=True)
 lib=build/'cargo/release/libtokenizers_c.a'
 subprocess.run(['cmake','-S',str(source),'-B',str(build/'cmake'),'-DCMAKE_BUILD_TYPE=Release','-DTOKENIZERS_LIBRARY='+str(lib)],check=True)
 subprocess.run(['cmake','--build',str(build/'cmake'),'--parallel','4'],check=True)
 suffix='dylib' if platform.system()=='Darwin' else 'so';encoder=build/'cmake'/('libpotasms_encoder.'+suffix);quantizer=build/'cmake/quantize'
 if not encoder.exists() or not quantizer.exists():raise RuntimeError('Native outputs missing')
 result={'runtime_source_sha256':spec['sha256'],'encoder_library':str(encoder.relative_to(work)),'encoder_sha256':digest(encoder),'quantizer':str(quantizer.relative_to(work)),'quantizer_sha256':digest(quantizer),'python_dependency':False,'backend':'CPU4/repack0/standardattention'};(build/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result
