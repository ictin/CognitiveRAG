from __future__ import annotations
import hashlib, json
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numcodecs
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

PROTOCOL='NECF-001-WP2-CONFIRMATORY-2024-v1'
PREFLIGHT_VERSION='B2-HRRR-SOURCE-ADAPTER-PREFLIGHT-v1'
SAMPLE_DATES=['2020-08-01','2021-07-16','2023-07-15','2024-07-15']
POINTS={
'CISO':[('Sacramento',38.5816,-121.4944),('Fresno',36.7378,-119.7871),('Los_Angeles',34.0522,-118.2437),('San_Diego',32.7157,-117.1611)],
'ERCO':[('Dallas_Fort_Worth',32.8998,-97.0403),('Houston',29.7604,-95.3698),('Austin',30.2672,-97.7431),('San_Antonio',29.4241,-98.4936)],
'PJM':[('Philadelphia',39.9526,-75.1652),('Washington_DC',38.9072,-77.0369),('Pittsburgh',40.4406,-79.9959),('Columbus',39.9612,-82.9988)],
'ISNE':[('Boston',42.3601,-71.0589),('Hartford',41.7658,-72.6734),('Providence',41.8240,-71.4128),('Manchester_NH',42.9956,-71.4548)],
}
X0=-2697520.1425219304; Y0=-1587306.1525566636; DX=3000.0; DY=3000.0
NX=1799; NY=1059
CRS_HRRR=CRS.from_proj4('+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +a=6371229 +b=6371229 +units=m +no_defs')
TRANSFORMER=Transformer.from_crs('EPSG:4326',CRS_HRRR,always_xy=True)
RETRY=Config(signature_version=UNSIGNED,retries={'max_attempts':10,'mode':'adaptive'},connect_timeout=10,read_timeout=60)
NOAA=boto3.client('s3',region_name='us-east-1',config=RETRY)
ZARR=boto3.client('s3',region_name='us-west-1',config=RETRY)


def point_index(lon,lat):
    x,y=TRANSFORMER.transform(lon,lat)
    ix=int(round((x-X0)/DX)); iy=int(round((y-Y0)/DY))
    if not (0<=ix<NX and 0<=iy<NY): raise ValueError((lat,lon,ix,iy))
    return {'ix':ix,'iy':iy,'chunk_id':f'{iy//150}.{ix//150}','in_x':ix%150,'in_y':iy%150}


def s3_json(client,bucket,key):
    return json.loads(client.get_object(Bucket=bucket,Key=key)['Body'].read())


def zarr_base(run,var):
    return run.strftime(f'sfc/%Y%m%d/%Y%m%d_%Hz_fcst.zarr/2m_above_ground/{var}/2m_above_ground/{var}')


def decode_chunk(run,var,chunk_id,meta,cache):
    key=(run.isoformat(),var,chunk_id)
    if key in cache:return cache[key]
    objkey=f'{zarr_base(run,var)}/0.{chunk_id}'
    payload=ZARR.get_object(Bucket='hrrrzarr',Key=objkey)['Body'].read()
    codec=numcodecs.get_codec(meta['compressor']) if meta.get('compressor') else None
    raw=codec.decode(payload) if codec else payload
    arr=np.frombuffer(raw,dtype=np.dtype(meta['dtype']))
    chunks=tuple(int(v) for v in meta['chunks'])
    if arr.size != int(np.prod(chunks)):
        raise RuntimeError(f'chunk-size mismatch {objkey}: decoded={arr.size} expected={np.prod(chunks)}')
    arr=arr.reshape(chunks)
    cache[key]=arr
    return arr


def run():
    all_point_meta={}
    for ba,pts in POINTS.items():
        all_point_meta[ba]=[]
        for name,lat,lon in pts:
            all_point_meta[ba].append({'name':name,'lat':lat,'lon':lon,**point_index(lon,lat)})
    result={'protocol_id':PROTOCOL,'preflight_version':PREFLIGHT_VERSION,'status':'PASS','source_adapter':'POINT_IN_TIME_ELIGIBLE_DERIVED_HRRR_MIRROR','point_index':all_point_meta,'samples':[]}
    cache={}
    for ds in SAMPLE_DATES:
        day=pd.Timestamp(ds)
        origin=pd.Timestamp(f'{ds} 10:30',tz='America/New_York').tz_convert('UTC')
        init=pd.Timestamp(f'{ds} 12:00',tz='UTC')
        valids=pd.date_range(origin.ceil('h'),origin+pd.Timedelta(hours=12),freq='h')
        fxx=[int((v-init)/pd.Timedelta(hours=1)) for v in valids]
        if len(fxx)!=12 or min(fxx)<1: raise RuntimeError(f'bad target leads {ds} {fxx}')
        heads=[]
        for fx in fxx:
            key=init.strftime('hrrr.%Y%m%d/conus/hrrr.t%Hz.wrfsfcf')+f'{fx:02d}.grib2'
            h=NOAA.head_object(Bucket='noaa-hrrr-bdp-pds',Key=key)
            lm=pd.Timestamp(h['LastModified']).tz_convert('UTC')
            heads.append({'fxx':fx,'key':key,'last_modified_utc':lm.isoformat(),'eligible':bool(lm<=origin),'content_length':int(h['ContentLength']),'etag':str(h.get('ETag','')).strip('"')})
        if not all(x['eligible'] for x in heads):
            result['status']='FAIL'; raise RuntimeError(f'NOAA availability gate failed {ds}')
        metas={}
        for var in ['TMP','DPT']:
            mk=f'{zarr_base(init,var)}/.zarray'
            meta=s3_json(ZARR,'hrrrzarr',mk)
            metas[var]={'dtype':meta['dtype'],'shape':meta['shape'],'chunks':meta['chunks'],'compressor':meta.get('compressor')}
        vals={'TMP':[],'DPT':[]}; missing=0
        for ba,pts in all_point_meta.items():
            for pt in pts:
                for var in ['TMP','DPT']:
                    meta=s3_json(ZARR,'hrrrzarr',f'{zarr_base(init,var)}/.zarray')
                    arr=decode_chunk(init,var,pt['chunk_id'],meta,cache)
                    for fx in fxx:
                        lead_index=fx-1
                        if lead_index>=arr.shape[0]: raise RuntimeError(f'lead out of range {ds} {var} f{fx} shape={arr.shape}')
                        value=float(arr[lead_index,pt['in_y'],pt['in_x']])
                        if np.isfinite(value): vals[var].append(value)
                        else: missing+=1
        tr=np.asarray(vals['TMP']); dr=np.asarray(vals['DPT'])
        plausible=bool(len(tr)==192 and len(dr)==192 and missing==0 and np.all((tr>180)&(tr<340)) and np.all((dr>150)&(dr<330)))
        if not plausible:
            result['status']='FAIL'; raise RuntimeError(f'physical/finite preflight failed {ds}')
        rawhash=hashlib.sha256(np.asarray(vals['TMP']+vals['DPT'],dtype=np.float64).tobytes()).hexdigest()
        result['samples'].append({'date':ds,'forecast_origin_utc':origin.isoformat(),'init_utc':init.isoformat(),'target_fxx':fxx,'original_noaa_all_objects_available_before_origin':True,'original_noaa_latest_required_object_utc':max(x['last_modified_utc'] for x in heads),'zarr_metadata':metas,'unique_spatial_chunks':sorted({p['chunk_id'] for v in all_point_meta.values() for p in v}),'finite_values':{'TMP':len(tr),'DPT':len(dr)},'plausible_ranges_K':{'TMP':[float(tr.min()),float(tr.max())],'DPT':[float(dr.min()),float(dr.max())]},'sample_values_sha256':rawhash})
    # Detect the expected metadata precision transition rather than assuming it.
    dtypes=[(s['date'],s['zarr_metadata']['TMP']['dtype'],s['zarr_metadata']['DPT']['dtype']) for s in result['samples']]
    result['dtype_observations']=dtypes
    Path('experiment_output').mkdir(exist_ok=True)
    Path('experiment_output/B2_HRRR_PREFLIGHT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': run()
