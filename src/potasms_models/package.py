from pathlib import Path
import base64,json,math,shutil
import numpy as np
from .data import digest
from .model import fingerprint

def files(folder):return {p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in folder.iterdir() if p.is_file() and p.name!='model.json'}

def character(work):
 work=Path(work);out=work/'packages/lightweight';out.mkdir(parents=True,exist_ok=False);shutil.copy2(work/'character/seed-42/base.v1.json',out/'base.v1.json');base=json.loads((out/'base.v1.json').read_text());m={'formatVersion':1,'displayName':'Lightweight','engine':'character-linear-int8-v1','featureContract':base['featureContract'],'dimensions':base['buckets'],'files':files(out),'scoring':{'thresholdLogit':math.log(base['threshold']/(1-base['threshold'])),'calibration':None},'training':{'recipe':'character-v1','primarySeed':42,'smsRevision':'sms-v1','privateData':False},'personalization':{'method':'PA-II-fixed-bias','privateByDefault':True}};m['modelFingerprint']=fingerprint(m);(out/'model.json').write_text(json.dumps(m,indent=2)+'\n');return out

def student(work):
 work=Path(work);source=work/'student-export';out=work/'packages/multilingual';out.mkdir(parents=True,exist_ok=False)
 for name in ['encoder.q8_0.gguf','encoder.q8_0.gguf.pooler.bin']:shutil.copy2(source/name,out/name)
 head=np.load(source/'head.npz');(out/'head.f32.bin').write_bytes(head['w'].astype('<f4').tobytes());cal=json.loads((source/'q8_0-calibration.json').read_text());m={'formatVersion':1,'displayName':'Multilingual','engine':'compact-bert-q8-v1','featureContract':'compact-bert-bpe32768-pooler128-v1','dimensions':128,'files':files(out),'scoring':{'bias':float(np.float32(head['bias'])),'calibration':cal['calibration'],'thresholdLogit':cal['point']['threshold']},'training':{'recipe':'distilled-sms-q8-v1','smsRevision':'sms-v1','semanticRevision':'semantic-v1','seed':17,'privateData':False},'personalization':{'method':'PA-II-fixed-bias','privateByDefault':True}};m['modelFingerprint']=fingerprint(m);(out/'model.json').write_text(json.dumps(m,indent=2)+'\n')
 # App-facing, typed extension of the existing model JSON. The GGUF stays separate.
 base={'formatVersion':2,'modelVersion':'potasms-multilingual-'+m['modelFingerprint'][:16],'featureContract':m['featureContract'],'weightScale':1.,'bias':m['scoring']['bias'],'threshold':float(1/(1+math.exp(-m['scoring']['thresholdLogit']))),'buckets':128,'minN':2,'maxN':5,'quantizedWeights':'','float32Weights':base64.b64encode((out/'head.f32.bin').read_bytes()).decode(),'decisionLogit':m['scoring']['thresholdLogit'],'scoreCalibration':cal['calibration'],'encoder':{'sha256':m['files']['encoder.q8_0.gguf']['sha256'],'bytes':m['files']['encoder.q8_0.gguf']['bytes'],'poolerSHA256':m['files']['encoder.q8_0.gguf.pooler.bin']['sha256'],'poolerBytes':m['files']['encoder.q8_0.gguf.pooler.bin']['bytes'],'maximumTokens':256},'attribution':['LEALLA-small Apache-2.0 initialization','BGE-M3 MIT semantic teacher','Tatoeba CC0/CC BY2.0FR; MASSIVE/Banking77/UCI/IMC2025 CC BY4.0; original fictional SMS fixtures']};(source/'base.v2.json').write_text(json.dumps(base,separators=(',',':'))+'\n');return out
