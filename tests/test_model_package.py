import base64,json,hashlib
import numpy as np
import pytest
from potasms_models.model import Model,fingerprint
from potasms_models import native

def package(tmp_path,bias=6.):
 tmp_path.mkdir();p=tmp_path/'base.v1.json';p.write_text(json.dumps({'quantizedWeights':base64.b64encode(bytes(1024)).decode(),'weightScale':1.,'bias':bias,'buckets':1024,'featureContract':native.LEGACY_CONTRACT,'threshold':.995}))
 m={'formatVersion':1,'engine':'character-linear-int8-v1','featureContract':native.LEGACY_CONTRACT,'dimensions':1024,'files':{'base.v1.json':{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}},'scoring':{'thresholdLogit':float(np.log(.995/.005)),'calibration':None}}
 m['modelFingerprint']=fingerprint(m);(tmp_path/'model.json').write_text(json.dumps(m));return tmp_path

def test_switching_models_cannot_reuse_another_models_personal_state(tmp_path):
 a=Model(package(tmp_path/'a'));b=Model(package(tmp_path/'b',-6.));state=a.empty()
 with pytest.raises(ValueError,match='another model'):b.predict('Fictional notice',state)
 assert a.predict('Fictional notice',state)['junk'] is True
 assert b.predict('Fictional notice')['junk'] is False

def test_cancel_after_an_update_does_not_mutate_the_source_state(tmp_path):
 model=Model(package(tmp_path/'a'));state=model.empty();calls=[0]
 def cancel():calls[0]+=1;return calls[0]>1
 with pytest.raises(InterruptedError):model.learn([('Fictional family dinner notice',0)],state,cancelled=cancel)
 assert calls[0]==2 and not state['delta'].any()
 changed=model.learn([('Fictional family dinner notice',0)],state)
 assert np.linalg.norm(changed['delta'])>0 and not state['delta'].any()
 assert model.predict('Fictional family dinner notice',changed)['decisionMargin']<model.predict('Fictional family dinner notice',state)['decisionMargin']

def test_tampered_payload_is_rejected_before_inference(tmp_path):
 folder=package(tmp_path/'a');(folder/'base.v1.json').write_text('{}')
 with pytest.raises(ValueError,match='payload'):Model(folder)
