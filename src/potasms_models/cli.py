from pathlib import Path
import argparse,json,os,subprocess,sys

def main():
 os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
 p=argparse.ArgumentParser(description='Reproducible public SMS filtering baselines')
 p.add_argument('--work',type=Path,default=Path('work/run-v1'))
 sub=p.add_subparsers(dest='command',required=True)
 prep=sub.add_parser('prepare');prep.add_argument('--semantic-archive',type=Path)
 fetch=sub.add_parser('fetch-models');fetch.add_argument('--model-cache',type=Path)
 runtime=sub.add_parser('build-runtime');runtime.add_argument('--runtime-archive',type=Path);runtime.add_argument('--offline',action='store_true')
 sub.add_parser('train-character');sub.add_parser('train-student');sub.add_parser('finalize-student');sub.add_parser('package');sub.add_parser('doctor')
 infer=sub.add_parser('infer');infer.add_argument('--model',type=Path,required=True);infer.add_argument('--text',required=True);infer.add_argument('--encoder-library',type=Path)
 a=p.parse_args();work=a.work.resolve();os.environ['POTASMS_WORK']=str(work)
 from .bootstrap import build_native,semantic,models
 library=build_native()
 if a.command=='prepare':
  from .data import prepare
  print(prepare(work));print(semantic(work,a.semantic_archive))
 elif a.command=='fetch-models':print(models(work,a.model_cache))
 elif a.command=='build-runtime':
  from .runtime import build
  print(json.dumps(build(work,a.runtime_archive,a.offline),indent=2))
 elif a.command=='train-character':
  from .character import train
  print(json.dumps(train(work),indent=2))
 elif a.command=='train-student':
  from .student_pipeline import train
  train(work)
 elif a.command=='finalize-student':
  from .student_finalize import finalize
  print(json.dumps(finalize(work),indent=2))
 elif a.command=='package':
  from .package import character,student
  print(character(work));print(student(work))
 elif a.command=='infer':
  from .model import Model
  with Model(a.model,a.encoder_library) as model:print(json.dumps(model.predict(a.text)))
 elif a.command=='doctor':
  import platform,importlib.metadata as m
  print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'native_library':str(library),'work':str(work),'packages':{k:m.version(k) for k in ['numpy','scipy','scikit-learn']}},indent=2))

if __name__=='__main__':main()
