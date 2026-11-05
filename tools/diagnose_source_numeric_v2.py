"""Investigate logged warnings with source-only replays, no target access."""
import os
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[k]="1"
import json
import warnings
import numpy as np
from nir_dg.preprocessing import make_preprocessor
from nir_dg.series_v2.data import source_real,features
from nir_dg.series_v2.engine import verify_freeze,folds,ROOT,write_json,now
from nir_dg.series_v2.models import Model,grids


if __name__=="__main__":
    protocol=verify_freeze()["protocol"]
    specs=[s for s in protocol["jobs"] if s["dataset"]=="appliances" and s["replica"]==0 and s["model_seed"]==11]
    rows=[];fits=0
    for spec in specs:
        frame=source_real(spec)
        for fi,(tr,va) in enumerate(folds(frame,spec)):
            processor=make_preprocessor(frame.iloc[tr][features(frame)])
            X=processor.fit_transform(frame.iloc[tr][features(frame)]);valid=processor.transform(frame.iloc[va][features(frame)])
            y=frame.iloc[tr].__y.to_numpy();env=frame.iloc[tr].__env.to_numpy()
            names=processor.get_feature_names_out()
            for ci,params in enumerate(grids("regression")["irf"]):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always",RuntimeWarning)
                    model=Model("irf","regression",11,params).fit(X,y,env)
                fits+=1;anomalies=[]
                def visit(node,ids,depth,tree):
                    if node.feature<0:
                        return
                    x=X[ids,node.feature];mask=x<=node.threshold
                    if mask.sum()<12 or (~mask).sum()<12:
                        unique=np.unique(x)
                        anomalies.append(dict(tree=tree,depth=depth,feature=str(names[node.feature]),threshold=node.threshold,parent_rows=len(ids),left_rows=int(mask.sum()),right_rows=int((~mask).sum()),unique_largest_values=[float(v) for v in unique[-3:]],threshold_equals_node_max=bool(node.threshold==x.max())))
                    visit(node.left,ids[mask],depth+1,tree);visit(node.right,ids[~mask],depth+1,tree)
                for ti,(tree,ids) in enumerate(zip(model.model_.trees_,model.model_.bootstrap_indices_)):
                    visit(tree,ids,0,ti)
                p=model.predict(valid)
                row=dict(job_id=spec["job_id"],k=spec["k"],n=spec["n"],fold=fi,config=ci,warnings=[str(w.message) for w in caught],split_anomalies=anomalies,validation_predictions_finite=bool(np.isfinite(p).all()))
                rows.append(row)
                print("SOURCE NUMERIC",spec["k"],spec["n"],fi,ci,"warnings",len(caught),"anomalies",len(anomalies),flush=True)
    write_json(ROOT/"reports/series_v2/source_numeric_diagnostic.json",dict(at=now(),source_only=True,diagnostic_extra_fits=fits,contexts=rows,frozen_runtime_modified=False,interpretation="Floating-point midpoint can round to the upper observed value for near-adjacent cyclic feature values; actual partitions can violate registered minimum leaf size. Observed finite prediction checks are recorded, not assumed. No corrective model decisions were evaluated on the opened holdout."))
