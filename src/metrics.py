"""
Common metrics module for production schedules.
Calculates the same KPIs for EDD, Energy-aware, and Slack-aware schedules.
"""
import pandas as pd
import json
from pathlib import Path

SLOT_MIN = 15
LOAD_CO2_FACTOR = {"Light": 0.35, "Medium": 0.45, "Maximum": 0.60}

def _to_dt(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df

def normalize_schedule(schedule, machines, tariff):
    """Ensure schedules from different schedulers have comparable energy/cost columns."""
    schedule = _to_dt(schedule, ['start_time', 'end_time', 'due_time'])
    tariff = _to_dt(tariff, ['time_slot'])
    machines = machines.copy()
    m_power = machines.set_index('machine_id')['operation_power_kW'].to_dict()

    energy_vals, cost_vals, peak_vals, co2_vals = [], [], [], []
    for _, r in schedule.iterrows():
        start, end = r['start_time'], r['end_time']
        load_type = r.get('load_type', 'Medium')
        power_kw = float(m_power.get(r['machine_id'], 0))
        duration_h = (end - start).total_seconds() / 3600

        # If schedule already has energy/cost, we still recompute using a common rule for fairness.
        slots = tariff[(tariff['time_slot'] >= start.floor('15min')) & (tariff['time_slot'] < end)]
        if slots.empty:
            kwh = power_kw * duration_h
            cost = kwh * 140
            peak = 0
        else:
            kwh_per_slot = power_kw * (SLOT_MIN / 60)
            kwh = len(slots) * kwh_per_slot
            cost = float((slots['price_per_kWh'] * kwh_per_slot).sum())
            peak = int(slots['peak_flag'].sum())
        co2 = kwh * LOAD_CO2_FACTOR.get(load_type, 0.45)
        energy_vals.append(round(kwh, 3))
        cost_vals.append(round(cost, 2))
        peak_vals.append(peak)
        co2_vals.append(round(co2, 3))

    schedule['common_energy_kWh'] = energy_vals
    schedule['common_energy_cost'] = cost_vals
    schedule['common_peak_overlap_slots'] = peak_vals
    schedule['common_co2_kg'] = co2_vals
    return schedule

def build_power_profile(schedule, machines, tariff):
    """Return 15-min slot power profile by summing operation power of active operations."""
    schedule = _to_dt(schedule, ['start_time', 'end_time'])
    tariff = _to_dt(tariff, ['time_slot'])
    m_power = machines.set_index('machine_id')['operation_power_kW'].to_dict()
    profile = tariff[['time_slot', 'tariff_type', 'price_per_kWh', 'peak_flag']].copy()
    profile['total_power_kW'] = 0.0
    for _, r in schedule.iterrows():
        mask = (profile['time_slot'] >= r['start_time'].floor('15min')) & (profile['time_slot'] < r['end_time'])
        profile.loc[mask, 'total_power_kW'] += float(m_power.get(r['machine_id'], 0))
    profile['energy_kWh'] = profile['total_power_kW'] * (SLOT_MIN / 60)
    profile['energy_cost'] = profile['energy_kWh'] * profile['price_per_kWh']
    return profile

def calculate_metrics(schedule, job_orders, machines, tariff, schedule_name='schedule'):
    job_orders = _to_dt(job_orders, ['release_time', 'due_time'])
    schedule = normalize_schedule(schedule, machines, tariff)
    completion = schedule.groupby('job_id')['end_time'].max()
    due = job_orders.set_index('job_id')['due_time'].to_dict()
    tardiness = completion.index.to_series().map(lambda j: max(0, (completion[j] - due[j]).total_seconds()/60))
    setup_col = 'setup_time_min' if 'setup_time_min' in schedule.columns else None
    profile = build_power_profile(schedule, machines, tariff)

    machine_util = []
    for mid, g in schedule.groupby('machine_id'):
        busy_h = ((g['end_time'] - g['start_time']).dt.total_seconds()/3600).sum()
        machine_util.append({'machine_id': mid, 'busy_hours': round(float(busy_h), 2)})

    metrics = {
        'schedule_name': schedule_name,
        'scheduled_operations': int(len(schedule)),
        'scheduled_jobs': int(completion.shape[0]),
        'on_time_jobs': int((tardiness == 0).sum()),
        'tardy_jobs': int((tardiness > 0).sum()),
        'due_date_adherence_rate': round(float((tardiness == 0).mean()), 4),
        'total_tardiness_min': round(float(tardiness.sum()), 2),
        'avg_tardiness_min': round(float(tardiness.mean()), 2),
        'makespan_start': str(schedule['start_time'].min()),
        'makespan_end': str(schedule['end_time'].max()),
        'total_setup_time_min': int(schedule[setup_col].sum()) if setup_col else 0,
        'total_energy_kWh': round(float(schedule['common_energy_kWh'].sum()), 3),
        'total_energy_cost': round(float(schedule['common_energy_cost'].sum()), 2),
        'total_peak_overlap_slots': int(schedule['common_peak_overlap_slots'].sum()),
        'peak_period_energy_kWh': round(float(profile.loc[profile['peak_flag'] == 1, 'energy_kWh'].sum()), 3),
        'max_demand_kW': round(float(profile['total_power_kW'].max()), 3),
        'total_co2_kg': round(float(schedule['common_co2_kg'].sum()), 3),
        'machine_busy_hours': machine_util,
    }
    return metrics, schedule, profile

def save_metrics(metrics, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
