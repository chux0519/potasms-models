"""48 fictional preference streams, identical feedback order for both engines."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .model import Model
from .data import digest

def evaluate(work):
 work=Path(work);out=work/'personal-learning-v1';out.mkdir(exist_ok=False)
 rows=[json.loads(line) for line in (work/'sms-v1/rows.jsonl').read_text().splitlines()]
 runtime=json.loads((work/'runtime-build/manifest.json').read_text());library=work/runtime['encoder_library'];summary={}
 for name in ['lightweight','multilingual']:
  started=time.perf_counter();records=[]
  with Model(work/'packages'/name,library) as model:
   # Representations are immutable. Memoization changes neither native arithmetic
   # nor examples/order/weights; it avoids repeated encoder execution per stream.
   encode=model._features;cache={}
   def features(text):
    if text not in cache:cache[text]=encode(text)
    return cache[text]
   model._features=features
   personal=[r for r in rows if r['pool']=='personal'];bg=[r for r in rows if r['pool']=='background']
   for r in personal+bg:features(r['text'])
   for topic in sorted({r['topic'] for r in personal}):
    for label in [0,1]:
     for source in ['en','zh']:
      candidates=[r for r in personal if r['topic']==topic and r['role']=='feedback' and r['language']==source]
      assert len(candidates)==10
      for seed in [17,42,73]:
       order=np.random.default_rng(seed).permutation(len(candidates));state=model.empty();pool=[]
       for n in range(11):
        if n:pool.append((candidates[order[n-1]]['text'],label));state=model.learn(pool,state)
        if n not in [0,1,5,10]:continue
        metrics={}
        for language in ['en','zh','mixed']:
         target=[r for r in personal if r['topic']==topic and r['role']=='novel' and r['language']==language]
         protected=[r for r in personal if r['topic']==topic and r['role']=='protected' and r['language']==language]
         metrics[language]={'target_correct':sum(model.predict(r['text'],state)['junk']==bool(label) for r in target),'target_total':len(target),'protected_fp':sum(model.predict(r['text'],state)['junk'] for r in protected),'protected_normal':len(protected)}
        normals=[r for r in bg if not r['label']];junk=[r for r in bg if r['label']]
        metrics['background']={'fp':sum(model.predict(r['text'],state)['junk'] for r in normals),'normal':len(normals),'tp':sum(model.predict(r['text'],state)['junk'] for r in junk),'junk':len(junk)}
        records.append(dict(topic=topic,label=label,source_language=source,arrival_seed=seed,n_labels=n,metrics=metrics))
   result={'modelFingerprint':model.identity,'streams':48,'records':records,'seconds':time.perf_counter()-started,'protocol':'warm replay all confirmed examples,24 epochs per arrival,PA-II C1/margin.05/clip2/norm12; fixed bias; no threshold retuning','synthetic':True,'independent_market_acceptance':False}
   for n in [0,1,5,10]:
    selected=[r for r in records if r['n_labels']==n]
    result.setdefault('summary',{})[str(n)]={
     'cross_language_preference_accuracy':float(np.mean([r['metrics']['zh' if r['source_language']=='en' else 'en']['target_correct']/6 for r in selected])),
     'protected_normal_fpr_for_junk_preferences':float(np.mean([r['metrics'][language]['protected_fp']/4 for r in selected if r['label']==1 for language in ['en','zh','mixed']])),
     'background_fp_mean':float(np.mean([r['metrics']['background']['fp'] for r in selected])),
     'background_tp_mean':float(np.mean([r['metrics']['background']['tp'] for r in selected])),
    }
   (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');summary[name]=result['summary'];print('PERSONAL_LEARNING_DONE',name,summary[name],flush=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
