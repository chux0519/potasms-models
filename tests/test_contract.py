import numpy as np
import pytest
from potasms_models.evaluation import common_language_threshold,score_metrics
from potasms_models.character import logits
from potasms_models import native
import base64

@pytest.mark.parametrize('bias,junk',[(-6.,False),(6.,True)])
def test_actual_native_probability_matches_logit_domain(bias,junk):
 base={'quantizedWeights':base64.b64encode(bytes(1024)).decode(),'weightScale':1.,'bias':bias,'featureContract':native.CONTRACT}
 raw=logits(base,[{'text':'Fictional threshold-domain test'}]);assert raw[0]==pytest.approx(bias)
 assert bool(raw[0]>=np.log(.995/.005))==junk
 assert bool(native.library().pota_sigmoid(raw[0])>=.995)==junk

def test_language_cap_is_selected_from_calibration_only():
 y=np.array([0,1]*3);z=np.array([2.,3.,5.,6.,8.,9.]);language=['en','en','zh','zh','mixed','mixed']
 result=common_language_threshold(y,z,language,.001)
 assert result['calibration_only'] and result['actual_holdout_fpr_guaranteed'] is False
 assert all(v['fp']==0 for v in result['panels'].values())
 # A future normal can exceed the calibration threshold; no heldout guarantee.
 assert score_metrics([0,1],[100.,101.],result['threshold'])['fp']==1

def test_unicode_feature_contracts_are_explicit():
 assert native.normalize('Straße',fold=True,contract=native.LEGACY_CONTRACT)=='straße'
 assert native.normalize('Straße',fold=True,contract=native.CONTRACT)=='strasse'
 with pytest.raises(ValueError):native.features('hello',contract='unknown')
