"""Rehydrate the exact frozen public membership; never scan a private directory."""
from pathlib import Path
import csv,hashlib,io,json,urllib.request,zipfile
from . import native
from .authored_sms import synthetic_rows
from .semantic_learn_data import build as personal_rows
from .decision_language_cases import authored_rows

ROOT=Path(__file__).resolve().parents[2]
SOURCES={
 'uci':dict(url='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',sha256='1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3',file='uci.zip',license='CC-BY-4.0',attribution='Almeida & Hidalgo, UCI SMS Spam Collection, DOI10.24432/C5CC84'),
 'imc25':dict(url='https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/a6175560b57387199871e51fbef6bc523d2516b4/dataset/final_dataset_output.csv',sha256='1bbd1e9e82c3ea023112207b80da268a5c4a07d2353c2b0898360ab037fa9a64',file='imc25.csv',license='CC-BY-4.0',attribution='Agarwal et al., Fishing for Smishing, IMC2025, DOI10.1145/3730567.3764431'),
}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def fetch(url,path,expected):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if digest(path)!=expected:raise ValueError('Existing download differs: '+str(path))
  return path
 temp=path.with_suffix(path.suffix+'.partial')
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'POTASMS-public-reproduction/0.1'}),timeout=120) as src,temp.open('xb') as out:
  while block:=src.read(1024*1024):out.write(block)
 if digest(temp)!=expected:raise ValueError('Upstream snapshot changed; download rejected: '+url)
 temp.rename(path);return path

def prepare(work):
 work=Path(work);target=work/'sms-v1';target.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((ROOT/'manifests/sms-membership-v1.json').read_text());entries=manifest['entries'];wanted={e['text_sha256'] for e in entries};texts={}
 def add(text):
  value=hashlib.sha256(text.encode()).hexdigest()
  if value in wanted:texts[value]=text
 for row in synthetic_rows()+personal_rows()+authored_rows():add(row['text'])
 for key,s in SOURCES.items():fetch(s['url'],work/'downloads'/s['file'],s['sha256'])
 with zipfile.ZipFile(work/'downloads/uci.zip') as z:
  for line in z.read('SMSSpamCollection').decode('utf8').splitlines():
   _,text=line.split('\t',1);add(native.normalize(text))
 with (work/'downloads/imc25.csv').open(newline='',encoding='utf8') as f:
  for row in csv.DictReader(f):
   try:add(native.normalize(row['text']))
   except ValueError:continue
 missing=wanted-texts.keys()
 if missing:raise ValueError(f'{len(missing)} frozen text hashes could not be rehydrated; no replacement or resampling allowed')
 rows=[{k:texts[e['text_sha256']] if k=='text' else e['metadata'][k] for k in e['keys']} for e in entries]
 output=target/'rows.jsonl';output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
 if digest(output)!=manifest['rows_sha256']:raise ValueError('Full row-file checksum differs from the published benchmark')
 (target/'sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
 (target/'manifest.json').write_text(json.dumps({'revision':'sms-v1','rows_sha256':digest(output),'rows':len(rows),'sources':SOURCES,'private_data':False,'synthetic_background':True,'background_scene_families':32},indent=2)+'\n')
 return output
