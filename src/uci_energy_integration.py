"""
UCI Steel Industry Energy Consumption Dataset integration helper.

Purpose
- Load the real UCI steel energy dataset when available.
- Build Load_Type-based reference profiles: mean power, CO2 factor/proxy, NSM/time profile.
- Recalculate each generated production schedule using UCI-derived Load_Type energy multipliers.

Expected input
- --uci-file path/to/Steel_industry_data.csv   (optional)
- If no UCI file is provided, this script creates a synthetic-compatible reference profile
  so the rest of the pipeline can still run and be replaced later by the real UCI data.

Outputs
- uci_load_type_profile.csv
- <schedule>_uci_enriched.csv
- uci_schedule_comparison.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

SLOT_MIN = 15
DEFAULT_CO2_FACTOR = {"Light": 0.35, "Medium": 0.45, "Maximum": 0.60}

LOAD_ALIASES = {
    'light_load': 'Light', 'light': 'Light', 'Light_Load': 'Light', 'Light': 'Light',
    'medium_load': 'Medium', 'medium': 'Medium', 'Medium_Load': 'Medium', 'Medium': 'Medium',
    'maximum_load': 'Maximum', 'maximum': 'Maximum', 'Maximum_Load': 'Maximum', 'Maximum': 'Maximum',
}

def norm_load(x):
    if pd.isna(x): return 'Medium'
    s = str(x).strip()
    return LOAD_ALIASES.get(s, LOAD_ALIASES.get(s.lower(), s))

def find_col(df, candidates):
    lower = {c.lower().replace(' ', '').replace('_',''): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('_','')
        if key in lower: return lower[key]
    return None

def load_uci(uci_file):
    df = pd.read_csv(uci_file)
    load_col = find_col(df, ['Load_Type','Load Type','load_type'])
    usage_col = find_col(df, ['Usage_kWh','Usage kWh','usage_kwh','Lagging_Current_Reactive.Power_kVarh'])
    co2_col = find_col(df, ['CO2(tCO2)','CO2','CO2_tCO2'])
    date_col = find_col(df, ['date','Date'])
    nsm_col = find_col(df, ['NSM'])
    if load_col is None:
        raise ValueError('UCI file must contain Load_Type column or equivalent.')
    if usage_col is None:
        raise ValueError('UCI file must contain Usage_kWh or equivalent power/energy column.')
    out = pd.DataFrame()
    out['load_type'] = df[load_col].map(norm_load)
    out['usage_kWh'] = pd.to_numeric(df[usage_col], errors='coerce')
    if co2_col:
        out['co2_raw'] = pd.to_numeric(df[co2_col], errors='coerce')
    else:
        out['co2_raw'] = np.nan
    if nsm_col:
        out['nsm'] = pd.to_numeric(df[nsm_col], errors='coerce')
        out['hour'] = (out['nsm'] // 3600).astype('Int64')
    elif date_col:
        dt = pd.to_datetime(df[date_col], errors='coerce')
        out['hour'] = dt.dt.hour
    else:
        out['hour'] = np.nan
    out = out.dropna(subset=['usage_kWh'])
    return out

def create_reference_profile(uci_file=None):
    if uci_file and Path(uci_file).exists():
        uci = load_uci(uci_file)
        base = uci.groupby('load_type', as_index=False).agg(
            uci_mean_usage_kWh=('usage_kWh','mean'),
            uci_median_usage_kWh=('usage_kWh','median'),
            uci_max_usage_kWh=('usage_kWh','max'),
            sample_count=('usage_kWh','count'),
            uci_mean_co2_raw=('co2_raw','mean')
        )
        # Convert raw CO2 to kg/kWh proxy. If UCI CO2 is unavailable/zero, fallback.
        vals=[]
        for _, r in base.iterrows():
            lt=r['load_type']; mean_usage=max(float(r['uci_mean_usage_kWh']), 1e-9)
            raw=r['uci_mean_co2_raw']
            if pd.notna(raw) and raw>0:
                # UCI column is often tCO2 per interval; convert t to kg and divide by kWh.
                cf=float(raw)*1000/mean_usage
                # Clamp unrealistic values for PoC stability.
                cf=min(max(cf, 0.05), 2.0)
            else:
                cf=DEFAULT_CO2_FACTOR.get(lt,0.45)
            vals.append(cf)
        base['co2_kg_per_kWh'] = vals
        # Relative multiplier vs Medium for schedule energy adjustment
        med = base.loc[base.load_type=='Medium','uci_mean_usage_kWh']
        med_val = float(med.iloc[0]) if len(med) else float(base.uci_mean_usage_kWh.mean())
        base['uci_energy_multiplier_vs_medium'] = base['uci_mean_usage_kWh'] / med_val
        source = 'real_uci'
    else:
        base = pd.DataFrame([
            {'load_type':'Light','uci_mean_usage_kWh':35,'uci_median_usage_kWh':35,'uci_max_usage_kWh':50,'sample_count':0,'uci_mean_co2_raw':np.nan,'co2_kg_per_kWh':0.35,'uci_energy_multiplier_vs_medium':0.50},
            {'load_type':'Medium','uci_mean_usage_kWh':75,'uci_median_usage_kWh':75,'uci_max_usage_kWh':100,'sample_count':0,'uci_mean_co2_raw':np.nan,'co2_kg_per_kWh':0.45,'uci_energy_multiplier_vs_medium':1.00},
            {'load_type':'Maximum','uci_mean_usage_kWh':140,'uci_median_usage_kWh':140,'uci_max_usage_kWh':180,'sample_count':0,'uci_mean_co2_raw':np.nan,'co2_kg_per_kWh':0.60,'uci_energy_multiplier_vs_medium':1.85},
        ])
        source = 'fallback_profile_replace_with_real_uci'
    base['profile_source'] = source
    return base

def enrich_schedule(schedule_path, machines, tariff, profile, out_dir):
    sched = pd.read_csv(schedule_path)
    sched['start_time'] = pd.to_datetime(sched['start_time']); sched['end_time'] = pd.to_datetime(sched['end_time'])
    tariff = tariff.copy(); tariff['time_slot'] = pd.to_datetime(tariff['time_slot'])
    m_power = machines.set_index('machine_id')['operation_power_kW'].to_dict()
    prof = profile.set_index('load_type').to_dict('index')
    rows=[]
    for _, r in sched.iterrows():
        lt = norm_load(r.get('load_type','Medium'))
        p = prof.get(lt, prof.get('Medium'))
        duration_h = (r.end_time-r.start_time).total_seconds()/3600
        base_kwh = float(m_power.get(r.machine_id,0))*duration_h
        multiplier = float(p['uci_energy_multiplier_vs_medium'])
        # Blend machine-rated energy and UCI load multiplier to avoid unrealistic explosion.
        uci_kwh = base_kwh * (0.7 + 0.3*multiplier)
        slots = tariff[(tariff.time_slot >= r.start_time.floor('15min')) & (tariff.time_slot < r.end_time)]
        avg_price = float(slots.price_per_kWh.mean()) if len(slots) else 140.0
        peak_slots = int(slots.peak_flag.sum()) if len(slots) else 0
        rows.append((round(uci_kwh,3), round(uci_kwh*avg_price,2), peak_slots, round(uci_kwh*float(p['co2_kg_per_kWh']),3)))
    sched[['uci_energy_kWh','uci_energy_cost','uci_peak_overlap_slots','uci_co2_kg']] = pd.DataFrame(rows, index=sched.index)
    out_path = out_dir / (Path(schedule_path).stem + '_uci_enriched.csv')
    sched.to_csv(out_path, index=False, encoding='utf-8-sig')
    return {
        'schedule_name': Path(schedule_path).stem,
        'uci_total_energy_kWh': round(float(sched.uci_energy_kWh.sum()),3),
        'uci_total_energy_cost': round(float(sched.uci_energy_cost.sum()),2),
        'uci_total_peak_overlap_slots': int(sched.uci_peak_overlap_slots.sum()),
        'uci_total_co2_kg': round(float(sched.uci_co2_kg.sum()),3),
        'enriched_file': out_path.name,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--uci-file', default=None)
    ap.add_argument('--data-dir', default='/home/user/input')
    ap.add_argument('--schedule-dir', default='/home/user/output')
    ap.add_argument('--output-dir', default='/home/user/output')
    args=ap.parse_args()
    data=Path(args.data_dir); sched_dir=Path(args.schedule_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    def d(original, fid):
        p1=data/original; p2=data/f'data_{fid}.csv'; return pd.read_csv(p1 if p1.exists() else p2)
    machines=d('machines.csv',7148716); tariff=d('tou_tariff.csv',7148722)
    profile=create_reference_profile(args.uci_file)
    profile.to_csv(out/'uci_load_type_profile.csv', index=False, encoding='utf-8-sig')
    schedules=['fifo_schedule.csv','edd_schedule.csv','energy_aware_schedule.csv','slack_aware_energy_schedule.csv','slack_aware_energy_v2_schedule.csv']
    summary=[]
    for s in schedules:
        p=sched_dir/s
        if p.exists(): summary.append(enrich_schedule(p,machines,tariff,profile,out))
    pd.DataFrame(summary).to_csv(out/'uci_schedule_comparison.csv', index=False, encoding='utf-8-sig')
    with open(out/'uci_integration_summary.json','w',encoding='utf-8') as f: json.dump({'uci_file':args.uci_file,'profile_rows':len(profile),'schedules':summary},f,ensure_ascii=False,indent=2)
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__': main()
