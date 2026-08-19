from __future__ import annotations
import argparse, hashlib, json, math, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROTOCOL_ID = 'NECF-001-WP2-CONFIRMATORY-2024-v1'
B1_RESULT_SHA256 = '3eadb09ad82ac5b5338c9dd26eadac592bd5378fb4139a1b7a0cc65cbdca1d89'
B1_OPERATOR_MAE = 1725.0005719842131
B1_ROUTED_MAE = 1272.9666006614848
SEED = 20260819
BOOT = 5000
WEATHER_COLS = [
    'hrrr_temperature_c', 'hrrr_dewpoint_c', 'hrrr_relative_humidity',
    'hrrr_heat_index_c', 'hrrr_cooling_degree_hours_18c'
]
POINTS={
'CISO':[('Sacramento',38.5816,-121.4944),('Fresno',36.7378,-119.7871),('Los_Angeles',34.0522,-118.2437),('San_Diego',32.7157,-117.1611)],
'ERCO':[('Dallas_Fort_Worth',32.8998,-97.0403),('Houston',29.7604,-95.3698),('Austin',30.2672,-97.7431),('San_Antonio',29.4241,-98.4936)],
'PJM':[('Philadelphia',39.9526,-75.1652),('Washington_DC',38.9072,-77.0369),('Pittsburgh',40.4406,-79.9959),('Columbus',39.9612,-82.9988)],
'ISNE':[('Boston',42.3601,-71.0589),('Hartford',41.7658,-72.6734),('Providence',41.8240,-71.4128),('Manchester_NH',42.9956,-71.4548)],
}
X0=-2697520.1425219304; Y0=-1587306.1525566636; DX=3000.0; DY=3000.0
NX=1799; NY=1059


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rh_from_t_td(t_c: float, td_c: float) -> float:
    a,b=17.625,243.04
    es_td=math.exp(a*td_c/(b+td_c)); es_t=math.exp(a*t_c/(b+t_c))
    return float(np.clip(100.0*es_td/es_t,0.0,100.0))


def hi_c(t_c: float, rh: float) -> float:
    t_f=t_c*9/5+32
    if t_f<80 or rh<40: return t_c
    hi=(-42.379+2.04901523*t_f+10.14333127*rh-0.22475541*t_f*rh
        -0.00683783*t_f*t_f-0.05481717*rh*rh+0.00122874*t_f*t_f*rh
        +0.00085282*t_f*rh*rh-0.00000199*t_f*t_f*rh*rh)
    return (hi-32)*5/9


def weather_runtime():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    import numcodecs
    from pyproj import CRS, Transformer
    cfg=Config(signature_version=UNSIGNED,retries={'max_attempts':12,'mode':'adaptive'},connect_timeout=10,read_timeout=90,max_pool_connections=80)
    noaa=boto3.client('s3',region_name='us-east-1',config=cfg)
    zarr=boto3.client('s3',region_name='us-west-1',config=cfg)
    crs=CRS.from_proj4('+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +a=6371229 +b=6371229 +units=m +no_defs')
    transformer=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    return noaa,zarr,numcodecs,transformer


def build_point_index(transformer):
    out={}
    for ba,pts in POINTS.items():
        out[ba]=[]
        for name,lat,lon in pts:
            x,y=transformer.transform(lon,lat)
            ix=int(round((x-X0)/DX)); iy=int(round((y-Y0)/DY))
            if not (0<=ix<NX and 0<=iy<NY): raise ValueError((name,lat,lon,ix,iy))
            out[ba].append({'name':name,'lat':lat,'lon':lon,'ix':ix,'iy':iy,'chunk_id':f'{iy//150}.{ix//150}','in_x':ix%150,'in_y':iy%150})
    return out


def zarr_base(run: pd.Timestamp, var: str) -> str:
    return run.strftime(f'sfc/%Y%m%d/%Y%m%d_%Hz_fcst.zarr/2m_above_ground/{var}/2m_above_ground/{var}')


def get_json(client,bucket,key):
    return json.loads(client.get_object(Bucket=bucket,Key=key)['Body'].read())


def decode_chunk(zarr, numcodecs, run, var, chunk_id, meta):
    key=f'{zarr_base(run,var)}/0.{chunk_id}'
    payload=zarr.get_object(Bucket='hrrrzarr',Key=key)['Body'].read()
    codec=numcodecs.get_codec(meta['compressor']) if meta.get('compressor') else None
    raw=codec.decode(payload) if codec else payload
    arr=np.frombuffer(raw,dtype=np.dtype(meta['dtype']))
    chunks=tuple(int(v) for v in meta['chunks'])
    expected=int(np.prod(chunks))
    if arr.size != expected:
        raise RuntimeError(f'decoded chunk size mismatch {key}: {arr.size} != {expected}')
    return arr.reshape(chunks)


def required_valids(day: str):
    origin=pd.Timestamp(f'{day} 10:30',tz='America/New_York').tz_convert('UTC')
    init=pd.Timestamp(f'{day} 12:00',tz='UTC')
    valids=pd.date_range(origin.ceil('h'),origin+pd.Timedelta(hours=12),freq='h')
    fxx=[int((v-init)/pd.Timedelta(hours=1)) for v in valids]
    if len(valids)!=12 or min(fxx)<1:
        raise RuntimeError(f'bad target window {day}: {fxx}')
    return origin,init,valids,fxx


def extract_weather_day(day: str, noaa, zarr, numcodecs, pindex):
    origin,init,valids,fxx=required_valids(day)
    heads=[]
    try:
        for fx in fxx:
            key=init.strftime('hrrr.%Y%m%d/conus/hrrr.t%Hz.wrfsfcf')+f'{fx:02d}.grib2'
            h=noaa.head_object(Bucket='noaa-hrrr-bdp-pds',Key=key)
            lm=pd.Timestamp(h['LastModified']).tz_convert('UTC')
            heads.append((fx,lm,key,int(h['ContentLength']),str(h.get('ETag','')).strip('"')))
        if not all(lm<=origin for _,lm,_,_,_ in heads):
            return [],{'day':day,'status':'SKIP','reason':'NOAA_REQUIRED_OBJECT_AFTER_ORIGIN','origin_utc':origin.isoformat(),'latest_object_utc':max(lm for _,lm,_,_,_ in heads).isoformat()}
        metas={}
        for var in ('TMP','DPT'):
            metas[var]=get_json(zarr,'hrrrzarr',f'{zarr_base(init,var)}/.zarray')
        chunk_ids=sorted({p['chunk_id'] for pts in pindex.values() for p in pts})
        arrays={}
        for var in ('TMP','DPT'):
            for cid in chunk_ids:
                arrays[(var,cid)]=decode_chunk(zarr,numcodecs,init,var,cid,metas[var])
        rows=[]
        for ba,pts in pindex.items():
            for valid,fx in zip(valids,fxx):
                derived=[]
                for p in pts:
                    ti=arrays[('TMP',p['chunk_id'])]
                    di=arrays[('DPT',p['chunk_id'])]
                    li=fx-1
                    if li>=ti.shape[0] or li>=di.shape[0]:
                        raise RuntimeError(f'lead out of range {day} f{fx} TMP{ti.shape} DPT{di.shape}')
                    tk=float(ti[li,p['in_y'],p['in_x']]); dk=float(di[li,p['in_y'],p['in_x']])
                    if not (np.isfinite(tk) and np.isfinite(dk) and 180<tk<340 and 150<dk<330):
                        raise RuntimeError(f'nonfinite/implausible weather {day} {ba} {p["name"]} f{fx}')
                    tc=tk-273.15; dc=dk-273.15; rh=rh_from_t_td(tc,dc); hi=hi_c(tc,rh)
                    derived.append((tc,dc,rh,hi,max(0.0,tc-18.0)))
                a=np.asarray(derived,float)
                rows.append({'datetime_utc':valid.isoformat(),'ba_code':ba,
                             'hrrr_temperature_c':float(a[:,0].mean()),
                             'hrrr_dewpoint_c':float(a[:,1].mean()),
                             'hrrr_relative_humidity':float(a[:,2].mean()),
                             'hrrr_heat_index_c':float(a[:,3].mean()),
                             'hrrr_cooling_degree_hours_18c':float(a[:,4].mean())})
        return rows,{'day':day,'status':'OK','origin_utc':origin.isoformat(),'init_utc':init.isoformat(),
                     'target_fxx':fxx,'latest_required_noaa_object_utc':max(lm for _,lm,_,_,_ in heads).isoformat(),
                     'noaa_availability_margin_minutes':float((origin-max(lm for _,lm,_,_,_ in heads))/pd.Timedelta(minutes=1)),
                     'zarr_dtype':{'TMP':metas['TMP']['dtype'],'DPT':metas['DPT']['dtype']},
                     'zarr_shape':{'TMP':metas['TMP']['shape'],'DPT':metas['DPT']['shape']},'rows':len(rows)}
    except Exception as e:
        return [],{'day':day,'status':'SKIP','reason':type(e).__name__,'message':str(e)[:800],'origin_utc':origin.isoformat(),'init_utc':init.isoformat()}


def cmd_extract_weather(args):
    noaa,zarr,numcodecs,transformer=weather_runtime(); pindex=build_point_index(transformer)
    days=[str(d.date()) for d in pd.date_range(args.start,args.end,freq='D')]
    rows=[]; records=[]
    workers=min(16,max(1,len(days)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(extract_weather_day,d,noaa,zarr,numcodecs,pindex):d for d in days}
        done=0
        for fut in as_completed(futs):
            r,m=fut.result(); rows.extend(r); records.append(m); done+=1
            if done%25==0 or done==len(days): print(f'WEATHER_PROGRESS {done}/{len(days)} rows={len(rows)}',flush=True)
    records.sort(key=lambda x:x['day'])
    w=pd.DataFrame(rows)
    if not w.empty:
        w['datetime_utc']=pd.to_datetime(w['datetime_utc'],utc=True)
        w=w.sort_values(['datetime_utc','ba_code']).drop_duplicates(['datetime_utc','ba_code'],keep='last')
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); w.to_csv(out,index=False)
    ok=sum(r['status']=='OK' for r in records); skip=len(records)-ok
    margins=[r['noaa_availability_margin_minutes'] for r in records if r.get('status')=='OK']
    manifest={'protocol_id':PROTOCOL_ID,'adapter':'POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR','start':args.start,'end':args.end,
              'days_total':len(records),'days_ok':ok,'days_skipped':skip,'rows':int(len(w)),
              'coverage_days':ok/len(records) if records else None,
              'availability_margin_minutes':{'min':float(min(margins)) if margins else None,'median':float(np.median(margins)) if margins else None,'max':float(max(margins)) if margins else None},
              'dtype_counts':{},'skipped':[r for r in records if r['status']!='OK'],'output':str(out),'output_sha256':sha256_file(out)}
    for r in records:
        if r.get('status')=='OK':
            k=f"TMP={r['zarr_dtype']['TMP']};DPT={r['zarr_dtype']['DPT']}"; manifest['dtype_counts'][k]=manifest['dtype_counts'].get(k,0)+1
    mp=Path(args.manifest); mp.parent.mkdir(parents=True,exist_ok=True); mp.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


def system_fit_predict(f: pd.DataFrame, features: list[str], b1mod):
    tr=b1mod.split(f,'2020-01-01T00:00:00Z','2022-12-31T23:00:00Z').dropna(subset=['demand','forecast','residual'])
    va=b1mod.split(f,'2023-01-01T00:00:00Z','2023-12-31T23:00:00Z').dropna(subset=['demand','forecast','residual'])
    te=b1mod.split(f,'2024-01-01T00:00:00Z','2024-12-31T23:00:00Z').dropna(subset=['demand','forecast','residual'])
    ridge,rm=b1mod.ridge_fit(tr[features],tr.residual,va[features],va.residual); hgb=b1mod.hgb_fit(tr[features],tr.residual)
    vr=va.forecast.to_numpy()+ridge.predict(va[features]); vh=va.forecast.to_numpy()+hgb.predict(va[features]); VP=np.column_stack([va.forecast.to_numpy(),vr,vh]); weights=b1mod.stack_fit(VP,va.demand.to_numpy())
    va=va.copy(); va['pred_stack']=VP@weights; wl=b1mod.weekly_lifts(va,'pred_stack'); router={}
    for ba in b1mod.BAS:
        ci=b1mod.boot([v for b,_,v in wl if b==ba]); router[ba]={'promoted':bool(ci['lower'] is not None and ci['lower']>0),'validation_BA_week_relative_lift_bootstrap':ci}
    trp=te.forecast.to_numpy()+ridge.predict(te[features]); thp=te.forecast.to_numpy()+hgb.predict(te[features]); tsp=np.column_stack([te.forecast.to_numpy(),trp,thp])@weights
    te=te.copy(); te['pred_ridge']=trp; te['pred_hgb']=thp; te['pred_stack']=tsp; te['router_promoted']=te.ba_code.map(lambda b:router[str(b)]['promoted']); te['pred_routed']=np.where(te.router_promoted,te.pred_stack,te.forecast)
    models={}
    for name,col in [('operator','forecast'),('ridge','pred_ridge'),('hgb','pred_hgb'),('stack','pred_stack'),('routed','pred_routed')]:
        models[name]=b1mod.metrics(te.demand,te[col],te.forecast if name!='operator' else None)
    return {'ridge':rm,'stack_weights':weights.tolist(),'router':router,'models':models,'n_train':len(tr),'n_validation':len(va),'n_holdout':len(te)},te


def week_pair_lift(df: pd.DataFrame, baseline_col: str, challenger_col: str):
    q=df.copy(); q['week']=pd.to_datetime(q.datetime_utc,utc=True).dt.tz_localize(None).dt.to_period('W').astype(str); rows=[]
    for (ba,w),g in q.groupby(['ba_code','week']):
        bm=float(np.mean(np.abs(g.demand-g[baseline_col]))); cm=float(np.mean(np.abs(g.demand-g[challenger_col])))
        if bm>0: rows.append({'ba_code':str(ba),'week':w,'relative_lift':(bm-cm)/bm,'baseline_mae':bm,'challenger_mae':cm})
    return pd.DataFrame(rows)


def cmd_score(args):
    import run_b1 as b1mod
    weather_files=sorted(Path(args.weather_dir).rglob('weather_*.csv'))
    weather_files=[p for p in weather_files if not p.name.endswith('_manifest.csv')]
    if not weather_files: raise RuntimeError('no weather CSV shards found')
    wf=[]
    for p in weather_files:
        q=pd.read_csv(p)
        if set(['datetime_utc','ba_code']).issubset(q.columns): wf.append(q)
    w=pd.concat(wf,ignore_index=True); w['datetime_utc']=pd.to_datetime(w.datetime_utc,utc=True); w=w.sort_values(['datetime_utc','ba_code']).drop_duplicates(['datetime_utc','ba_code'],keep='last')
    raw=b1mod.load_data(); good,bad=b1mod.quarantine(raw); f=b1mod.target_filter(b1mod.features(good))
    F1=b1mod.cols(f); b1res,b1te=system_fit_predict(f,F1,b1mod)
    if abs(b1res['models']['operator']['mae']-B1_OPERATOR_MAE)>1e-8 or abs(b1res['models']['routed']['mae']-B1_ROUTED_MAE)>1e-6:
        raise RuntimeError(f'B1 reproduction mismatch: {b1res["models"]["operator"]["mae"]} {b1res["models"]["routed"]["mae"]}')
    fw=f.merge(w,on=['datetime_utc','ba_code'],how='left')
    F2=F1+WEATHER_COLS
    b2res,b2te=system_fit_predict(fw,F2,b1mod)
    keys=['datetime_utc','ba_code']; comp=b1te[keys+['demand','forecast','pred_routed']].rename(columns={'pred_routed':'pred_b1'}).merge(b2te[keys+['pred_routed','router_promoted']].rename(columns={'pred_routed':'pred_b2','router_promoted':'b2_router_promoted'}),on=keys,how='inner',validate='one_to_one')
    if len(comp)!=len(b1te) or len(comp)!=len(b2te): raise RuntimeError('B1/B2 holdout row alignment mismatch')
    mae1=float(np.mean(np.abs(comp.demand-comp.pred_b1))); mae2=float(np.mean(np.abs(comp.demand-comp.pred_b2))); aggregate=(mae1-mae2)/mae1
    weekly=week_pair_lift(comp,'pred_b1','pred_b2'); ci=b1mod.boot(weekly.relative_lift.to_numpy())
    ba_gate={}; positive_median=0
    for ba,g in weekly.groupby('ba_code'):
        med=float(g.relative_lift.median()); mean=float(g.relative_lift.mean()); bic=b1mod.boot(g.relative_lift.to_numpy());
        if med>0: positive_median+=1
        cg=comp[comp.ba_code==ba]; bm=float(np.mean(np.abs(cg.demand-cg.pred_b1))); cm=float(np.mean(np.abs(cg.demand-cg.pred_b2)))
        ba_gate[str(ba)]={'median_weekly_improvement_vs_B1':med,'mean_weekly_improvement_vs_B1':mean,'bootstrap':bic,'B1_mae':bm,'B2_mae':cm,'relative_mae_improvement_vs_B1':(bm-cm)/bm,'B2_router_promoted':bool(cg.b2_router_promoted.iloc[0])}
    unrouted_degradations=[max(0.0,-v['relative_mae_improvement_vs_B1']) for v in ba_gate.values() if not v['B2_router_promoted']]
    max_unrouted=max(unrouted_degradations) if unrouted_degradations else 0.0
    gate={'aggregate_relative_mae_improvement_vs_B1':aggregate,'required_min':0.02,
          'BA_week_bootstrap':ci,'required_bootstrap_lower_gt':0.0,'BAs_positive_median_improvement':positive_median,'required_min_BAs_positive_median':3,
          'max_unrouted_BA_degradation':max_unrouted,'allowed_max_unrouted_BA_degradation':0.05}
    gate['pass']=bool(aggregate>=0.02 and ci['lower'] is not None and ci['lower']>0 and positive_median>=3 and max_unrouted<=0.05)
    cov={}
    for label,a,b in [('train','2020-01-01T00:00:00Z','2022-12-31T23:00:00Z'),('validation','2023-01-01T00:00:00Z','2023-12-31T23:00:00Z'),('confirmatory','2024-01-01T00:00:00Z','2024-12-31T23:00:00Z')]:
        s=b1mod.split(fw,a,b); complete=s[WEATHER_COLS].notna().all(axis=1); cov[label]={'rows':len(s),'weather_complete_rows':int(complete.sum()),'coverage':float(complete.mean()) if len(s) else None}
    result={'protocol_id':PROTOCOL_ID,'status':'FROZEN_B2_2024_RESULT','source_adapter':'POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR','B1_result_sha256':B1_RESULT_SHA256,'B1_reproduction':b1res,'B2':b2res,'weather_coverage':cov,'incremental_gate_vs_B1':gate,'by_ba_incremental_vs_B1':ba_gate,'n_quarantined':len(bad)}
    outdir=Path(args.out_dir); outdir.mkdir(parents=True,exist_ok=True); rp=outdir/'B2_2024.json'; rp.write_text(json.dumps(result,indent=2))
    comp.to_csv(outdir/'B2_2024_comparison_predictions.csv.gz',index=False,compression='gzip')
    manifests=[]
    for mp in sorted(Path(args.weather_dir).rglob('weather_*_manifest.json')):
        try: manifests.append(json.loads(mp.read_text()))
        except Exception: pass
    ps={'adapter':'POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR','shards':manifests,'weather_rows_combined':int(len(w)),'weather_combined_sha256':hashlib.sha256(w.to_csv(index=False).encode()).hexdigest()}
    (outdir/'B2_WEATHER_PROVENANCE_SUMMARY.json').write_text(json.dumps(ps,indent=2))
    verdict='PASS' if gate['pass'] else 'FAIL'
    lines=[f'# B2 2024 Confirmatory Verdict — {verdict}','',f'Protocol: `{PROTOCOL_ID}`','',f'B1 routed MAE: **{mae1:.3f} MW**',f'B2 routed MAE: **{mae2:.3f} MW**',f'Incremental aggregate MAE improvement vs frozen B1: **{aggregate*100:.3f}%**',f'Paired BA-week bootstrap 95% CI: **[{ci["lower"]*100:.3f}%, {ci["upper"]*100:.3f}%]**',f'BAs with positive median weekly improvement: **{positive_median}/4**',f'Maximum degradation among B2-unrouted BAs: **{max_unrouted*100:.3f}%**','',f'**Frozen gate verdict: {verdict}.**','', 'No 2025 replication data was used. The EIA historical source remains revision-sensitive; HRRR numeric values were accessed through the frozen derived Zarr mirror while original NOAA object timestamps governed point-in-time eligibility.']
    (outdir/'B2_2024_VERDICT.md').write_text('\n'.join(lines)+'\n')
    print('PFSCE_B2_2024_RESULT_BEGIN'); print(json.dumps(result,indent=2)); print('PFSCE_B2_2024_RESULT_END')


def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    ew=sp.add_parser('extract-weather'); ew.add_argument('--start',required=True); ew.add_argument('--end',required=True); ew.add_argument('--out',required=True); ew.add_argument('--manifest',required=True); ew.set_defaults(func=cmd_extract_weather)
    sc=sp.add_parser('score'); sc.add_argument('--weather-dir',required=True); sc.add_argument('--out-dir',default='experiment_output'); sc.set_defaults(func=cmd_score)
    a=ap.parse_args(); a.func(a)

if __name__=='__main__': main()
