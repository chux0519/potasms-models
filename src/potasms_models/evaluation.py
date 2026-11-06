"""Frozen calibration-only operating points; never compare probabilities to logits."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
LANGUAGES=['en','zh','mixed']

def calibrate_scores(y,z):
    y=np.asarray(y);z=np.asarray(z,dtype=float)
    if not np.isfinite(z).all() or set(y)!={0,1}:raise ValueError('Invalid calibration inputs')
    center=float(z.mean());scale=max(float(z.std()),1e-8);x=(z-center)/scale
    def loss(v):
        s=np.exp(v[0])*x+v[1]
        return float(np.mean(np.logaddexp(0,s)-y*s)+1e-4*(v[0]**2+v[1]**2))
    fit=minimize(loss,[0.,0.],bounds=[(-4,4),(-10,10)],method='L-BFGS-B')
    if not fit.success:raise ValueError(str(fit.message))
    return {'center':center,'scale':scale,'log_slope':float(fit.x[0]),'intercept':float(fit.x[1]),
            'only_calibration':True,'domain':'full finite logits, positive affine Platt scaling'}


def apply_scores(z,c):return np.exp(c['log_slope'])*(np.asarray(z,dtype=float)-c['center'])/c['scale']+c['intercept']


def choose_score_threshold(y,z,fpr):
    y=np.asarray(y);z=np.asarray(z,dtype=float);normal=y==0;junk=y==1
    assert normal.any() and junk.any() and np.isfinite(z).all()
    best=None
    for t in np.r_[np.unique(z),np.nextafter(z.max(),np.inf)]:
        pred=z>=t;fp=int((pred&normal).sum());tp=int((pred&junk).sum())
        if fp/normal.sum()>fpr+1e-12:continue
        key=(tp/junk.sum(),tp/max(tp+fp,1),float(t))
        if best is None or key>best[0]:best=(key,float(t))
    assert best is not None
    return best[1]


def common_language_threshold(y,logits,languages,fpr=.001):
    """One global calibration-only threshold, each language under the same empirical cap."""
    y=np.asarray(y);z=np.asarray(logits,dtype=float);language=np.asarray(languages)
    if y.ndim!=1 or z.shape!=y.shape or language.shape!=y.shape or not np.isfinite(z).all() or not 0<=fpr<1:raise ValueError('Invalid calibration arrays')
    if not set(y)<= {0,1} or set(language)!=set(LANGUAGES):raise ValueError('Expected three languages and binary labels')
    for name in LANGUAGES:
        if set(y[language==name])!={0,1}:raise ValueError('Both classes required per calibration language')
    sentinel=np.nextafter(z.max(),np.inf)
    if not np.isfinite(sentinel):raise ValueError('No finite all-allow threshold')
    best=None
    for t in np.r_[np.unique(z),sentinel]:
        pred=z>=t;panels={}
        for name in LANGUAGES:
            mask=language==name;n=mask&(y==0);j=mask&(y==1)
            panels[name]=dict(normal=int(n.sum()),junk=int(j.sum()),fp=int((pred&n).sum()),tp=int((pred&j).sum()))
        if any(v['fp']/v['normal']>fpr+1e-12 for v in panels.values()):continue
        key=(sum(v['tp']/v['junk'] for v in panels.values())/3,-sum(v['fp'] for v in panels.values()),float(t))
        if best is None or key>best[0]:best=(key,float(t),panels)
    assert best is not None
    return dict(threshold=best[1],domain='finite continuous score/logit',panels=best[2],calibration_only=True,
                one_global_threshold=True,actual_holdout_fpr_guaranteed=False,target_empirical_fpr=fpr)

def choose_threshold(y_true: np.ndarray, probability: np.ndarray, max_fpr: float = 0.01) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, p, drop_intermediate=False)
    feasible = np.flatnonzero(false_positive_rate <= max_fpr)
    if not len(feasible):
        return float(np.nextafter(np.max(p), np.inf))
    best_recall = np.max(true_positive_rate[feasible])
    best = feasible[true_positive_rate[feasible] == best_recall]
    # Of thresholds with equal recall, use the least conservative one. ROC's
    # first threshold may be infinity when accepting nothing is the only safe
    # choice; that result is valid and explicit rather than silently using 0.5.
    return float(np.min(thresholds[best]))



def score_metrics(y,z,t):
 y=np.asarray(y);z=np.asarray(z,dtype=float)
 if not np.isfinite(z).all():raise ValueError('Nonfinite logit')
 normal=y==0;junk=y==1;pred=z>=t;fp=int((pred&normal).sum());tp=int((pred&junk).sum());tn=int((~pred&normal).sum());fn=int((~pred&junk).sum())
 return dict(n=len(y),normal=int(normal.sum()),junk=int(junk.sum()),fp=fp,tp=tp,tn=tn,fn=fn,accuracy=(tp+tn)/len(y),recall=tp/int(junk.sum()) if junk.any() else None,fpr=fp/int(normal.sum()) if normal.any() else None,threshold=float(t),threshold_domain='logit',roc_auc=float(roc_auc_score(y,z)) if normal.any() and junk.any() else None,pr_auc=float(average_precision_score(y,z)) if normal.any() and junk.any() else None)

def panels(rows,score,threshold):
 y=np.array([r['label'] for r in rows]);out={}
 for pool in ['background','old-cohort','old-language']:
  mask=np.array([r['pool']==pool for r in rows]);out[pool]=score_metrics(y[mask],score[mask],threshold)
 for lang in LANGUAGES:
  mask=np.array([r['pool']=='background' and r.get('language')==lang for r in rows]);out['background-'+lang]=score_metrics(y[mask],score[mask],threshold)
 return out
