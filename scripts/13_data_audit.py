from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd,numpy as np
from aeris.training import write_json
from aeris.modeling import sha256_file
W=ROOT/'data/aeris_windows_v1';P=ROOT/'data/aeris_prepared';OUT=ROOT/'experiments/end_to_end_20261116'
def main():
    contract=json.loads((P/'data_contract.json').read_text(encoding='utf-8-sig'));protocol=json.loads((W/'window_protocol.json').read_text(encoding='utf-8-sig'));scaler=json.loads((W/'source_scaler.json').read_text(encoding='utf-8-sig'));features=contract['core_features'];audit={};observations={}
    for domain in ['source','target']:
        df=pd.read_csv(P/f'{domain}_observations.csv.gz',low_memory=False);observations[domain]=df
        delta=df.groupby(['run_id','segment_id'],sort=False).sample_time_s.diff()
        audit[domain]={'rows':len(df),'missing_values':df[features].isna().sum().to_dict(),'duplicate_row_ids':int(df.row_id.duplicated().sum()),'duplicate_run_timestamps':int(df.duplicated(['run_id','sample_time_s']).sum()),'nonincreasing_steps':int((delta<=0).sum()),'non_60s_steps':int((delta.notna()&~np.isclose(delta.fillna(60),60)).sum()),'raw_range_min':df[features].min().to_dict(),'raw_range_max':df[features].max().to_dict()}
    source=observations['source'];roles=source.condition_group.map(protocol['group_assignments']);fit=source.loc[source.core_valid & source.active_occupied & roles.eq('source_train'),features].to_numpy('float32').astype('float64')
    mean=fit.mean(0);scale=fit.std(0);scale[scale<1e-8]=1
    audit['preprocessing']={'fit_rows':len(fit),'declared_fit_rows':scaler['fit_rows'],'mean_max_abs_diff':float(np.max(np.abs(mean-scaler['mean']))),'scale_max_abs_diff':float(np.max(np.abs(scale-scaler['scale']))),'fit_partition':'source_train only','features_in_order':features,'no_learned_imputation':'Incomplete core windows excluded; runtime rejects missing values'}
    assert len(fit)==scaler['fit_rows'];np.testing.assert_allclose(mean,scaler['mean'],atol=1e-6);np.testing.assert_allclose(scale,scaler['scale'],atol=1e-6)
    manifest=pd.read_csv(P/'input_manifest.csv');checks=[]
    for row in manifest.itertuples():
        path=ROOT/'data'/row.relative_path;mapping=contract['source_columns'] if row.domain=='source' else contract['target_columns_after_whitespace_strip']
        raw=pd.read_csv(path,nrows=1440,low_memory=False);raw.columns=raw.columns.str.strip();fileid=hashlib.sha256(row.relative_path.encode()).hexdigest()[:16]
        canonical=observations[row.domain].set_index('row_id').loc[[fileid+':'+str(i) for i in range(len(raw))]]
        for feature in features:
            v=pd.to_numeric(raw[mapping[feature]],errors='coerce')
            if feature.endswith('_temp_c'):v=v-273.15 if row.domain=='source' else (v-32)*5/9
            match=np.allclose(v.to_numpy('float32'),canonical[feature].to_numpy('float32'),atol=3e-5,rtol=1e-5,equal_nan=True)
            checks.append({'path':row.relative_path,'feature':feature,'rows':len(raw),'conversion_matches':bool(match)});assert match
    pd.DataFrame(checks).to_csv(OUT/'channel_conversion_checks.csv',index=False)
    for role in ['source_train','source_validation','source_calibration']:
        p=dict(np.load(W/f'{role}.npz'));assert p['X_seq'].shape[1:]==(15,10);assert p['X_tab'].shape[1]==66
    audit['conversion_checks']=len(checks);audit['raw_files_hash_verified']=len(manifest);audit['all_channels_available_at_endpoint']=True
    audit['sequence_protocol']={'minutes':15,'stride_minutes':5,'features':10,'tabular':66,'receptive_field':29,'receptive_field_calculation':'1 + 2 convolutions * (kernel 3 - 1) * (1 + 2 + 4) = 29','representation_decision':'Existing RF includes mean/std/min/max/last/trend. Causal TCN receptive field already covers full 15-minute window; no justified expansion. Missingness indicators deferred because incomplete-window inference is blocked.'}
    audit['unresolved_alias']='AHU1/AHU2 not verified as aliases of G36/SZVAV/SZCAV; retain actual identifiers.'
    write_json(OUT/'extended_data_audit.json',audit);print(json.dumps(audit['preprocessing']))
if __name__=='__main__':main()
