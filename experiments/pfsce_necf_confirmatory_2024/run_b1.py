from __future__ import annotations
import hashlib, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROTOCOL_ID='NECF-001-WP2-CONFIRMATORY-2024-v1'
ORIGINAL_PROTOCOL_TREE_SHA256='40f1912ec2a95a91891400efdf0444e7aad67b444a19898b4d7294b5c198f28a'
PREREG_SHA256='ffccd7789cda190e10e7540f604f935d883faebecb1818dd20df9ec68d333da7'
ZENODO_RECORD='14881638'
ZENODO_CREATED='2025-02-17'
ARCHIVES={
'eia930-2020half1.zip':'354a1cda5b568fb34fea6d4ffd38fb28','eia930-2020half2.zip':'4012ed27c60582232348c3f19250441f',
'eia930-2021half1.zip':'8f56adfa0206e68932a8317c41d8dcdb','eia930-2021half2.zip':'7d949e9a6f017b09dd0a272c07e621f6',
'eia930-2022half1.zip':'915340b2854763dd2fe9e1bb93a630a5','eia930-2022half2.zip':'b3963a209ea8b5377e514feefb7aab67',
'eia930-2023half1.zip':'f58ba96b7c728c96f2df71541c7ab4ed','eia930-2023half2.zip':'c3cf3aae1bb657ca520689d528282608',
'eia930-2024half1.zip':'9d4bed4b10df2a2aa9093a24c8f960f6','eia930-2024half2.zip':'aca8e45d9b4486afe16688c0a08ff383'}
BAS=['CISO','ERCO','PJM','ISNE']; SEED=20260819; BOOT=5000


def digest(path, algo):
    h=hashlib.new(algo)
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def download(url,dest):
    if dest.exists(): return
    for i in range(7):
        try:
            with requests.get(url,stream=True,timeout=(30,180),headers={'User-Agent':'PFSCE-NECF-validation/1.0.1'}) as r:
                if r.status_code in {429,500,502,503,504}: raise RuntimeError(f'HTTP {r.status_code}')
                r.raise_for_status(); tmp=dest.with_suffix('.part')
                with open(tmp,'wb') as f:
                    for c in r.iter_content(1024*1024):
                        if c: f.write(c)
                tmp.replace(dest); return
        except Exception as e:
            if i==6: raise
            print('RETRY',dest.name,i+1,repr(e),flush=True); time.sleep(min(60,2**(i+1)))


def canon(c): return str(c).replace('\ufeff','').strip().lower().replace('_',' ')
def pick(headers):
    n={canon(c):c for c in headers}; out={}
    for target,names in {'ba':['ba','balancing authority'],'datetime_utc':['utc time','datetime utc','utc timestamp'],'forecast':['df','demand forecast'],'demand':['d','demand']}.items():
        for name in names:
            if name in n: out[target]=n[name]; break
        if target not in out: raise ValueError(f'missing {target}; headers={headers[:40]}')
    return out


def read_zip(path):
    frames=[]; meta=[]
    with zipfile.ZipFile(path) as z:
        ms=[m for m in z.namelist() if m.lower().endswith('.csv') and not m.endswith('/')]
        if not ms: raise RuntimeError(f'no csv in {path}')
        for m in ms:
            with z.open(m) as f: h=pd.read_csv(f,nrows=0)
            mp=pick(list(h.columns)); wanted=set(mp.values())
            with z.open(m) as f: r=pd.read_csv(f,usecols=lambda c:c in wanted,low_memory=False)
            r=r.rename(columns={v:k for k,v in mp.items()}); r['ba']=r['ba'].astype(str).str.strip(); r=r[r.ba.isin(BAS)].copy()
            if r.empty: meta.append({'member':m,'rows':0}); continue
            r['datetime_utc']=pd.to_datetime(r.datetime_utc,errors='coerce',utc=True)
            r['demand']=pd.to_numeric(r.demand,errors='coerce'); r['forecast']=pd.to_numeric(r.forecast,errors='coerce')
            r=r[['datetime_utc','ba','demand','forecast']].dropna(subset=['datetime_utc']); frames.append(r)
            meta.append({'member':m,'rows':len(r),'columns':mp})
    return pd.concat(frames,ignore_index=True),meta


def load_data():
    cache=Path('experiment_output/raw'); cache.mkdir(parents=True,exist_ok=True); frames=[]; provenance=[]
    for fn,md5 in ARCHIVES.items():
        p=cache/fn; url=f'https://zenodo.org/records/{ZENODO_RECORD}/files/{fn}?download=1'; print('DOWNLOAD',fn,flush=True); download(url,p)
        got=digest(p,'md5')
        if got!=md5: raise RuntimeError(f'MD5 mismatch {fn} expected={md5} got={got}')
        d,m=read_zip(p); frames.append(d); provenance.append({'file':fn,'md5':got,'sha256':digest(p,'sha256'),'size':p.stat().st_size,'members':m}); time.sleep(1)
    x=pd.concat(frames,ignore_index=True).sort_values(['ba','datetime_utc']).drop_duplicates(['ba','datetime_utc'],keep='last')
    Path('experiment_output').mkdir(exist_ok=True); x.to_csv('experiment_output/eia_2020_2024.csv',index=False)
    json.dump({'record':ZENODO_RECORD,'created':ZENODO_CREATED,'authority':'V1_PROVISIONAL_REVISION_SENSITIVE','files':provenance,'rows':len(x)},open('experiment_output/source_provenance.json','w'),indent=2)
    return x


def quarantine(x):
    ratio=x.demand/x.forecast; ok=x.demand.notna()&x.forecast.notna()&(x.demand>0)&(x.forecast>0)&ratio.between(.2,5)
    return x.loc[ok].copy(),x.loc[~ok].copy()


def regularize(x):
    fs=[]
    for ba,g in x.sort_values(['ba','datetime_utc']).drop_duplicates(['ba','datetime_utc'],keep='last').groupby('ba'):
        idx=pd.date_range(g.datetime_utc.min().floor('h'),g.datetime_utc.max().ceil('h'),freq='h',tz='UTC',name='datetime_utc')
        r=g.set_index('datetime_utc').reindex(idx); r['ba']=ba; fs.append(r.reset_index())
    return pd.concat(fs,ignore_index=True).sort_values(['ba','datetime_utc']).reset_index(drop=True)


def features(x):
    x=regularize(x); x['ba_code']=x.ba.astype(str); x['residual']=x.demand-x.forecast; dt=x.datetime_utc
    x['hour']=dt.dt.hour; x['day_of_week']=dt.dt.dayofweek; x['month']=dt.dt.month; x['is_weekend']=(dt.dt.dayofweek>=5).astype(int); g=x.groupby('ba',group_keys=False)
    for lag in [48,72,168]: x[f'residual_lag_{lag}']=g.residual.shift(lag); x[f'demand_lag_{lag}']=g.demand.shift(lag)
    for win in [168,336,720]:
        mp=max(24,win//4); x[f'resid_mean_{win}']=g.residual.transform(lambda s:s.shift(48).rolling(win,min_periods=mp).mean()); x[f'resid_std_{win}']=g.residual.transform(lambda s:s.shift(48).rolling(win,min_periods=mp).std())
    x=pd.get_dummies(x,columns=['ba'],prefix='ba',dtype=float); return x


def target_filter(x):
    t=x.datetime_utc.dt.tz_convert('America/New_York'); dates=t.dt.strftime('%Y-%m-%d'); origin=pd.to_datetime(dates+' 10:30',format='%Y-%m-%d %H:%M').dt.tz_localize('America/New_York'); delta=t-origin
    return x.loc[(delta>pd.Timedelta(0))&(delta<=pd.Timedelta(hours=12))].copy()


def split(x,a,b): return x[(x.datetime_utc>=pd.Timestamp(a))&(x.datetime_utc<=pd.Timestamp(b))].copy()
def cols(x): return [c for c in x.columns if c.startswith(('residual_lag_','demand_lag_','resid_mean_','resid_std_','ba_')) and c!='ba_code']+['forecast','hour','day_of_week','month','is_weekend']


def ridge_fit(trX,try_,vX,vy):
    best=None
    for a in [.1,1.,10.,100.]:
        m=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('ridge',Ridge(alpha=a))]); m.fit(trX,try_); mae=mean_absolute_error(vy,m.predict(vX))
        if best is None or mae<best[0]: best=(mae,a,m)
    return best[2],{'alpha':best[1],'validation_mae':best[0]}


def hgb_fit(X,y):
    m=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('hgb',HistGradientBoostingRegressor(learning_rate=.05,max_iter=300,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1.,random_state=SEED))]); return m.fit(X,y)


def stack_fit(P,y):
    n=P.shape[1]; res=minimize(lambda w:np.mean(np.abs(y-P@w)),np.repeat(1/n,n),method='SLSQP',bounds=[(0.,1.)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1.})
    if not res.success: raise RuntimeError(res.message)
    return res.x


def weekly_lifts(df,pred):
    q=df.copy(); q['week']=q.datetime_utc.dt.tz_localize(None).dt.to_period('W').astype(str); rows=[]
    for (ba,w),g in q.groupby(['ba_code','week']):
        b=np.mean(np.abs(g.demand-g.forecast)); m=np.mean(np.abs(g.demand-g[pred]));
        if b>0: rows.append((str(ba),w,(b-m)/b))
    return rows


def boot(vals):
    v=np.asarray(vals,float); v=v[np.isfinite(v)]
    if len(v)==0:return {'mean':None,'median':None,'lower':None,'upper':None,'n_clusters':0}
    rng=np.random.default_rng(SEED); means=np.array([rng.choice(v,size=len(v),replace=True).mean() for _ in range(BOOT)])
    return {'mean':float(v.mean()),'median':float(np.median(v)),'lower':float(np.quantile(means,.025)),'upper':float(np.quantile(means,.975)),'n_clusters':len(v)}


def metrics(y,p,b=None):
    o={'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5)}
    if b is not None: bm=float(mean_absolute_error(y,b)); o['baseline_mae']=bm; o['relative_mae_improvement']=(bm-o['mae'])/bm
    return o


def run():
    raw=load_data(); good,bad=quarantine(raw); f=target_filter(features(good)); F=cols(f)
    tr=split(f,'2020-01-01T00:00:00Z','2022-12-31T23:00:00Z').dropna(subset=['demand','forecast','residual']); va=split(f,'2023-01-01T00:00:00Z','2023-12-31T23:00:00Z').dropna(subset=['demand','forecast','residual']); te=split(f,'2024-01-01T00:00:00Z','2024-12-31T23:00:00Z').dropna(subset=['demand','forecast','residual'])
    ridge,rm=ridge_fit(tr[F],tr.residual,va[F],va.residual); hgb=hgb_fit(tr[F],tr.residual)
    vr=va.forecast.to_numpy()+ridge.predict(va[F]); vh=va.forecast.to_numpy()+hgb.predict(va[F]); VP=np.column_stack([va.forecast.to_numpy(),vr,vh]); w=stack_fit(VP,va.demand.to_numpy()); va=va.copy(); va['pred_stack']=VP@w
    wl=weekly_lifts(va,'pred_stack'); router={}
    for ba in BAS:
        ci=boot([v for b,_,v in wl if b==ba]); router[ba]={'promoted':bool(ci['lower'] is not None and ci['lower']>0),'validation_BA_week_relative_lift_bootstrap':ci}
    trp=te.forecast.to_numpy()+ridge.predict(te[F]); thp=te.forecast.to_numpy()+hgb.predict(te[F]); tsp=np.column_stack([te.forecast.to_numpy(),trp,thp])@w; te=te.copy(); te['pred_ridge']=trp; te['pred_hgb']=thp; te['pred_stack']=tsp; te['router_promoted']=te.ba_code.map(lambda b:router[str(b)]['promoted']); te['pred_routed']=np.where(te.router_promoted,te.pred_stack,te.forecast)
    result={'protocol_id':PROTOCOL_ID,'original_protocol_tree_sha256':ORIGINAL_PROTOCOL_TREE_SHA256,'preregistration_sha256':PREREG_SHA256,'implementation_correction':'v1.0.1 pre-holdout','source_vintage':{'zenodo_record':ZENODO_RECORD,'created':ZENODO_CREATED,'authority':'V1_PROVISIONAL_REVISION_SENSITIVE'},'n_raw':len(raw),'n_quarantined':len(bad),'n_train':len(tr),'n_validation':len(va),'n_holdout':len(te),'ridge':rm,'stack_weights':w.tolist(),'router':router,'models':{}}
    for name,col in [('operator','forecast'),('ridge','pred_ridge'),('hgb','pred_hgb'),('stack','pred_stack'),('routed','pred_routed')]: result['models'][name]=metrics(te.demand,te[col],te.forecast if name!='operator' else None)
    result['by_ba']={}
    for ba,g in te.groupby('ba_code'):
        br={'router_promoted':router[str(ba)]['promoted']}
        for name,col in [('operator','forecast'),('ridge','pred_ridge'),('hgb','pred_hgb'),('stack','pred_stack'),('routed','pred_routed')]: br[name]=metrics(g.demand,g[col],g.forecast if name!='operator' else None)
        br['holdout_BA_week_routed_lift_bootstrap']=boot([v for _,_,v in weekly_lifts(g,'pred_routed')]); result['by_ba'][str(ba)]=br
    allwl=weekly_lifts(te,'pred_routed'); result['BA_week_routed_lift_bootstrap']=boot([v for _,_,v in allwl]); result['primary_verdict_inputs']={'aggregate_relative_mae_improvement_vs_operator':result['models']['routed']['relative_mae_improvement'],'bootstrap_lower':result['BA_week_routed_lift_bootstrap']['lower'],'promoted_BAs':[b for b,r in router.items() if r['promoted']]}
    Path('experiment_output').mkdir(exist_ok=True); json.dump(result,open('experiment_output/B1_2024.json','w'),indent=2); te.to_csv('experiment_output/B1_2024_predictions.csv.gz',index=False,compression='gzip')
    print('PFSCE_B1_2024_RESULT_BEGIN'); print(json.dumps(result,indent=2)); print('PFSCE_B1_2024_RESULT_END')

if __name__=='__main__': run()
