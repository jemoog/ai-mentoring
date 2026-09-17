"""
EDD + Energy-aware scheduler.
Baseline order remains Earliest Due Date, but each operation is assigned to the
feasible machine/time candidate with the lowest weighted score:
  tardiness + energy cost + peak overlap + setup + CO2 proxy
"""
import argparse, json, math
from pathlib import Path
import pandas as pd

SLOT_MIN = 15
WEIGHTS = {
    "tardiness_min": 1000.0,
    "energy_cost": 1.0,
    "peak_slot": 500.0,
    "setup_min": 5.0,
    "co2_kg": 0.5,
    "completion_delay_min": 0.05,  # prevents unnecessary extreme delaying
}
LOAD_CO2_FACTOR = {"Light": 0.35, "Medium": 0.45, "Maximum": 0.60}  # kgCO2/kWh proxy

def ceil_to_slot(ts, slot_min=SLOT_MIN):
    ts = pd.Timestamp(ts)
    discard = pd.Timedelta(minutes=ts.minute % slot_min, seconds=ts.second, microseconds=ts.microsecond)
    floored = ts - discard
    return floored if floored == ts else floored + pd.Timedelta(minutes=slot_min)

def parse_hhmm(day, hhmm):
    h, m = map(int, str(hhmm).split(':'))
    if str(hhmm) == '23:59':
        return pd.Timestamp(day.date()) + pd.Timedelta(days=1)
    return pd.Timestamp(day.date()) + pd.Timedelta(hours=h, minutes=m)

def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end

def interval_conflict(start, end, intervals):
    return any(overlaps(start, end, s, e) for s, e in intervals)

def is_within_daily_availability(start, end, machine_row):
    cur = start
    while cur < end:
        day_start = pd.Timestamp(cur.date())
        avs = parse_hhmm(day_start, machine_row['available_start_time'])
        ave = parse_hhmm(day_start, machine_row['available_end_time'])
        seg_start = max(cur, day_start)
        seg_end = min(end, day_start + pd.Timedelta(days=1))
        if seg_start < avs or seg_end > ave:
            return False
        cur = day_start + pd.Timedelta(days=1)
    return True

def compute_duration_min(quantity, route_row, machine_row):
    capacity_time = quantity / machine_row['capacity_per_hour'] * 60
    return int(math.ceil(max(route_row['process_time_min'], capacity_time) / SLOT_MIN) * SLOT_MIN)

def tariff_stats(start, end, tariff):
    # Count 15-min tariff slots overlapping [start,end). Cost assumes kW * 0.25h per slot.
    slots = tariff[(tariff.time_slot >= start.floor('15min')) & (tariff.time_slot < end)]
    if slots.empty:
        return 0.0, 0, 0
    return float(slots.price_per_kWh.mean()), int(slots.peak_flag.sum()), len(slots)

def operation_cost(start, end, power_kw, load_type, tariff):
    slots = tariff[(tariff.time_slot >= start.floor('15min')) & (tariff.time_slot < end)]
    if slots.empty:
        kwh = power_kw * ((end-start).total_seconds()/3600)
        return kwh * 140, 0, kwh, kwh * LOAD_CO2_FACTOR.get(load_type, 0.45)
    kwh_per_slot = power_kw * (SLOT_MIN/60)
    cost = float((slots.price_per_kWh * kwh_per_slot).sum())
    peak_slots = int(slots.peak_flag.sum())
    kwh = float(len(slots) * kwh_per_slot)
    co2 = kwh * LOAD_CO2_FACTOR.get(load_type, 0.45)
    return cost, peak_slots, kwh, co2

def feasible_candidates(machine_id, duration_min, earliest_start, machines, machine_busy, maintenance, max_candidates=24, search_days=30):
    mrow = machines.loc[machines.machine_id == machine_id].iloc[0]
    t = ceil_to_slot(earliest_start)
    horizon_end = pd.Timestamp(machines.attrs['horizon_end'])
    search_end = min(horizon_end, t + pd.Timedelta(days=search_days))
    found=[]
    while t < search_end and len(found) < max_candidates:
        end = t + pd.Timedelta(minutes=duration_min)
        if is_within_daily_availability(t, end, mrow):
            intervals = machine_busy.get(machine_id, []) + maintenance.get(machine_id, [])
            if not interval_conflict(t, end, intervals):
                found.append((t,end))
        t += pd.Timedelta(minutes=SLOT_MIN)
    return found

def build_schedule(job_orders, machines, routing, maint, tariff, weights=WEIGHTS):
    for col in ['release_time','due_time']:
        job_orders[col] = pd.to_datetime(job_orders[col])
    maint['maintenance_start'] = pd.to_datetime(maint['maintenance_start'])
    maint['maintenance_end'] = pd.to_datetime(maint['maintenance_end'])
    tariff['time_slot'] = pd.to_datetime(tariff['time_slot'])
    horizon_end = max(job_orders.due_time.max(), maint.maintenance_end.max()) + pd.Timedelta(days=30)
    machines.attrs['horizon_end'] = str(horizon_end)

    machine_busy = {m: [] for m in machines.machine_id}
    maintenance = {m: [] for m in machines.machine_id}
    for _, r in maint.iterrows():
        maintenance[r.machine_id].append((r.maintenance_start, r.maintenance_end))
    last_product = {m: None for m in machines.machine_id}
    rows=[]

    jobs = job_orders.sort_values(['due_time','priority','release_time','job_id']).reset_index(drop=True)
    for _, job in jobs.iterrows():
        current_ready = job.release_time
        for step in sorted(routing[routing.product_type==job.product_type].process_step.unique()):
            eligible = routing[(routing.product_type==job.product_type) & (routing.process_step==step)]
            cand=[]
            for _, eroute in eligible.iterrows():
                mid = eroute.eligible_machine_id
                mrow = machines.loc[machines.machine_id==mid].iloc[0]
                setup_min = int(eroute.setup_time_min) if last_product[mid] is not None and last_product[mid] != job.product_type else 0
                proc_min = compute_duration_min(job.quantity, eroute, mrow)
                total_min = setup_min + proc_min
                for start,end in feasible_candidates(mid,total_min,current_ready,machines,machine_busy,maintenance):
                    setup_end = start + pd.Timedelta(minutes=setup_min)
                    # setup energy assumed 20% of machine operation power
                    setup_cost, setup_peak, setup_kwh, setup_co2 = operation_cost(start, setup_end, mrow.operation_power_kW*0.2, eroute.load_type, tariff) if setup_min else (0,0,0,0)
                    proc_cost, proc_peak, proc_kwh, proc_co2 = operation_cost(setup_end, end, mrow.operation_power_kW, eroute.load_type, tariff)
                    tardiness = max(0, (end - job.due_time).total_seconds()/60)
                    delay = max(0, (start - current_ready).total_seconds()/60)
                    total_cost = setup_cost + proc_cost
                    total_peak = setup_peak + proc_peak
                    total_kwh = setup_kwh + proc_kwh
                    total_co2 = setup_co2 + proc_co2
                    score = (weights['tardiness_min']*tardiness + weights['energy_cost']*total_cost +
                             weights['peak_slot']*total_peak + weights['setup_min']*setup_min +
                             weights['co2_kg']*total_co2 + weights['completion_delay_min']*delay)
                    cand.append((score, tardiness, end, start, mid, eroute, setup_min, proc_min, total_min, total_cost, total_peak, total_kwh, total_co2))
            if not cand:
                raise RuntimeError(f'No feasible candidate for job={job.job_id}, product={job.product_type}, step={step}')
            # hard preference: if any candidate completes by job due date, choose among on-time only.
            ontime = [c for c in cand if c[1] == 0]
            chosen = sorted(ontime if ontime else cand, key=lambda x: (x[0], x[2], x[3]))[0]
            score,tard,end,start,mid,eroute,setup_min,proc_min,total_min,cost,peak,kwh,co2 = chosen
            machine_busy[mid].append((start,end)); machine_busy[mid].sort()
            last_product[mid] = job.product_type
            current_ready = end
            rows.append({
                'job_id': job.job_id, 'product_type': job.product_type, 'quantity': int(job.quantity),
                'priority': int(job.priority), 'due_time': job.due_time, 'process_step': int(eroute.process_step),
                'process_id': eroute.process_id, 'machine_id': mid, 'load_type': eroute.load_type,
                'start_time': start, 'end_time': end, 'setup_time_min': setup_min,
                'processing_time_min': proc_min, 'total_time_min': total_min,
                'energy_kWh_est': round(kwh,3), 'energy_cost_est': round(cost,2),
                'peak_overlap_slots': int(peak), 'co2_kg_est': round(co2,3), 'candidate_score': round(score,3)
            })
    return pd.DataFrame(rows)

def metrics(schedule, job_orders):
    due = job_orders.assign(due_time=pd.to_datetime(job_orders.due_time)).set_index('job_id')['due_time'].to_dict()
    schedule['start_time']=pd.to_datetime(schedule.start_time); schedule['end_time']=pd.to_datetime(schedule.end_time)
    completion = schedule.groupby('job_id')['end_time'].max()
    tardiness = completion.index.to_series().map(lambda j: max(0,(completion[j]-due[j]).total_seconds()/60))
    return {
        'scheduled_operations': int(len(schedule)),
        'scheduled_jobs': int(completion.shape[0]),
        'on_time_jobs': int((tardiness==0).sum()),
        'tardy_jobs': int((tardiness>0).sum()),
        'due_date_adherence_rate': round(float((tardiness==0).mean()),4),
        'total_tardiness_min': round(float(tardiness.sum()),2),
        'avg_tardiness_min': round(float(tardiness.mean()),2),
        'makespan_start': str(schedule.start_time.min()),
        'makespan_end': str(schedule.end_time.max()),
        'total_setup_time_min': int(schedule.setup_time_min.sum()),
        'total_energy_kWh_est': round(float(schedule.energy_kWh_est.sum()),3),
        'total_energy_cost_est': round(float(schedule.energy_cost_est.sum()),2),
        'total_peak_overlap_slots': int(schedule.peak_overlap_slots.sum()),
        'total_co2_kg_est': round(float(schedule.co2_kg_est.sum()),3)
    }

def load_csv(inp, original, data_id):
    return pd.read_csv(inp/f'data_{data_id}.csv') if (inp/f'data_{data_id}.csv').exists() else pd.read_csv(inp/original)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='/home/user/input')
    ap.add_argument('--output-dir', default='/home/user/output')
    args=ap.parse_args()
    inp=Path(args.input_dir); out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    job_orders=load_csv(inp,'job_orders.csv',7148714)
    machines=load_csv(inp,'machines.csv',7148716)
    maint=load_csv(inp,'maintenance_schedule.csv',7148718)
    routing=load_csv(inp,'product_routing.csv',7148720)
    tariff=load_csv(inp,'tou_tariff.csv',7148722)
    sched=build_schedule(job_orders,machines,routing,maint,tariff)
    sched.to_csv(out/'energy_aware_schedule.csv', index=False, encoding='utf-8-sig')
    m=metrics(sched, job_orders)
    with open(out/'energy_aware_metrics.json','w',encoding='utf-8') as f: json.dump(m,f,ensure_ascii=False,indent=2)
    print(json.dumps(m,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
