"""
Slack-aware EDD + Energy-aware scheduler.

Goal:
- Keep EDD due-date priority as the main rule.
- Apply energy/peak/cost optimization only within a safe slack window.
- Avoid pushing urgent jobs too far into the future.

Outputs:
- slack_aware_energy_schedule.csv
- slack_aware_energy_metrics.json
"""
import argparse, json, math
from pathlib import Path
import pandas as pd

SLOT_MIN = 15

WEIGHTS = {
    "tardiness_min": 5000.0,      # stronger than v1
    "energy_cost": 1.0,
    "peak_slot": 700.0,
    "setup_min": 8.0,
    "co2_kg": 0.5,
    "delay_min": 10.0,           # prevents excessive delaying
    "completion_time_min": 0.2,   # favors early completion when scores are close
}

LOAD_CO2_FACTOR = {"Light": 0.35, "Medium": 0.45, "Maximum": 0.60}
LOAD_PEAK_MULTIPLIER = {"Light": 0.5, "Medium": 1.0, "Maximum": 2.0}


def ceil_to_slot(ts, slot_min=SLOT_MIN):
    ts = pd.Timestamp(ts)
    discard = pd.Timedelta(minutes=ts.minute % slot_min, seconds=ts.second, microseconds=ts.microsecond)
    floored = ts - discard
    return floored if floored == ts else floored + pd.Timedelta(minutes=slot_min)


def parse_hhmm(day, hhmm):
    hhmm = str(hhmm)
    h, m = map(int, hhmm.split(':'))
    if hhmm == '23:59':
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


def operation_cost(start, end, power_kw, load_type, tariff):
    if end <= start:
        return 0.0, 0, 0.0, 0.0
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


def slack_policy(slack_min, load_type):
    """Return candidate search window and mode based on slack."""
    if slack_min < 0:
        return 2, 8, "late_risk_fast"       # already risky: minimize delay
    if slack_min < 6*60:
        return 2, 8, "urgent_fast"          # almost EDD
    if slack_min < 24*60:
        return 6, 16, "limited_energy"      # short-range energy choice
    if load_type == "Maximum":
        return 24, 48, "active_peak_avoid"  # wider search for high energy process
    return 12, 32, "normal_energy"


def feasible_candidates(machine_id, duration_min, earliest_start, machines, machine_busy, maintenance, window_hours, max_candidates):
    mrow = machines.loc[machines.machine_id == machine_id].iloc[0]
    t = ceil_to_slot(earliest_start)
    horizon_end = pd.Timestamp(machines.attrs['horizon_end'])
    search_end = min(horizon_end, t + pd.Timedelta(hours=window_hours))
    found = []
    while t < search_end and len(found) < max_candidates:
        end = t + pd.Timedelta(minutes=duration_min)
        if is_within_daily_availability(t, end, mrow):
            intervals = machine_busy.get(machine_id, []) + maintenance.get(machine_id, [])
            if not interval_conflict(t, end, intervals):
                found.append((t, end))
        t += pd.Timedelta(minutes=SLOT_MIN)
    # fallback: if no candidate in slack window, search forward until one is found
    while not found and t < horizon_end:
        end = t + pd.Timedelta(minutes=duration_min)
        if is_within_daily_availability(t, end, mrow):
            intervals = machine_busy.get(machine_id, []) + maintenance.get(machine_id, [])
            if not interval_conflict(t, end, intervals):
                found.append((t, end))
                break
        t += pd.Timedelta(minutes=SLOT_MIN)
    return found


def build_schedule(job_orders, machines, routing, maint, tariff, weights=WEIGHTS):
    for col in ['release_time', 'due_time']:
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
    rows = []

    jobs = job_orders.sort_values(['due_time', 'priority', 'release_time', 'job_id']).reset_index(drop=True)

    for _, job in jobs.iterrows():
        current_ready = job.release_time
        product_routes = routing[routing.product_type == job.product_type]
        for step in sorted(product_routes.process_step.unique()):
            eligible = routing[(routing.product_type == job.product_type) & (routing.process_step == step)]
            all_candidates = []
            for _, eroute in eligible.iterrows():
                mid = eroute.eligible_machine_id
                mrow = machines.loc[machines.machine_id == mid].iloc[0]
                setup_min = int(eroute.setup_time_min) if last_product[mid] is not None and last_product[mid] != job.product_type else 0
                proc_min = compute_duration_min(job.quantity, eroute, mrow)
                total_min = setup_min + proc_min

                # approximate slack remaining for this operation; protects downstream steps imperfectly but usefully
                slack_min = (job.due_time - current_ready).total_seconds()/60 - total_min
                window_hours, max_cands, mode = slack_policy(slack_min, eroute.load_type)
                candidates = feasible_candidates(mid, total_min, current_ready, machines, machine_busy, maintenance, window_hours, max_cands)

                for start, end in candidates:
                    setup_end = start + pd.Timedelta(minutes=setup_min)
                    setup_cost, setup_peak, setup_kwh, setup_co2 = operation_cost(start, setup_end, mrow.operation_power_kW*0.2, eroute.load_type, tariff)
                    proc_cost, proc_peak, proc_kwh, proc_co2 = operation_cost(setup_end, end, mrow.operation_power_kW, eroute.load_type, tariff)
                    tardiness = max(0, (end - job.due_time).total_seconds()/60)
                    delay = max(0, (start - current_ready).total_seconds()/60)
                    completion_from_ready = max(0, (end - current_ready).total_seconds()/60)
                    cost = setup_cost + proc_cost
                    peak = setup_peak + proc_peak
                    kwh = setup_kwh + proc_kwh
                    co2 = setup_co2 + proc_co2
                    peak_multiplier = LOAD_PEAK_MULTIPLIER.get(eroute.load_type, 1.0)

                    # urgent jobs: strongly prefer earliest completion; energy matters only as tie-breaker
                    if mode in ["urgent_fast", "late_risk_fast"]:
                        score = (weights['tardiness_min']*tardiness +
                                 weights['completion_time_min']*50*completion_from_ready +
                                 weights['delay_min']*20*delay +
                                 weights['energy_cost']*0.2*cost +
                                 weights['peak_slot']*0.2*peak*peak_multiplier +
                                 weights['setup_min']*setup_min)
                    else:
                        score = (weights['tardiness_min']*tardiness +
                                 weights['energy_cost']*cost +
                                 weights['peak_slot']*peak*peak_multiplier +
                                 weights['setup_min']*setup_min +
                                 weights['co2_kg']*co2 +
                                 weights['delay_min']*delay +
                                 weights['completion_time_min']*completion_from_ready)

                    all_candidates.append({
                        'score': score, 'tardiness': tardiness, 'end': end, 'start': start,
                        'machine_id': mid, 'route': eroute, 'setup_min': setup_min,
                        'proc_min': proc_min, 'total_min': total_min, 'cost': cost,
                        'peak': peak, 'kwh': kwh, 'co2': co2, 'slack_min': slack_min,
                        'mode': mode
                    })
            if not all_candidates:
                raise RuntimeError(f'No feasible candidate for job={job.job_id}, product={job.product_type}, step={step}')

            ontime = [c for c in all_candidates if c['tardiness'] == 0]
            pool = ontime if ontime else all_candidates
            chosen = sorted(pool, key=lambda c: (c['score'], c['end'], c['start']))[0]

            start, end, mid = chosen['start'], chosen['end'], chosen['machine_id']
            machine_busy[mid].append((start, end)); machine_busy[mid].sort()
            last_product[mid] = job.product_type
            current_ready = end
            eroute = chosen['route']

            rows.append({
                'job_id': job.job_id,
                'product_type': job.product_type,
                'quantity': int(job.quantity),
                'priority': int(job.priority),
                'due_time': job.due_time,
                'process_step': int(eroute.process_step),
                'process_id': eroute.process_id,
                'machine_id': mid,
                'load_type': eroute.load_type,
                'start_time': start,
                'end_time': end,
                'setup_time_min': int(chosen['setup_min']),
                'processing_time_min': int(chosen['proc_min']),
                'total_time_min': int(chosen['total_min']),
                'energy_kWh_est': round(float(chosen['kwh']), 3),
                'energy_cost_est': round(float(chosen['cost']), 2),
                'peak_overlap_slots': int(chosen['peak']),
                'co2_kg_est': round(float(chosen['co2']), 3),
                'slack_before_operation_min': round(float(chosen['slack_min']), 2),
                'slack_policy_mode': chosen['mode'],
                'candidate_score': round(float(chosen['score']), 3)
            })
    return pd.DataFrame(rows)


def compute_metrics(schedule, job_orders):
    job_orders = job_orders.copy()
    job_orders['due_time'] = pd.to_datetime(job_orders['due_time'])
    schedule = schedule.copy()
    schedule['start_time'] = pd.to_datetime(schedule['start_time'])
    schedule['end_time'] = pd.to_datetime(schedule['end_time'])
    due = job_orders.set_index('job_id')['due_time'].to_dict()
    completion = schedule.groupby('job_id')['end_time'].max()
    tardiness = completion.index.to_series().map(lambda j: max(0, (completion[j]-due[j]).total_seconds()/60))
    return {
        'scheduled_operations': int(len(schedule)),
        'scheduled_jobs': int(completion.shape[0]),
        'on_time_jobs': int((tardiness == 0).sum()),
        'tardy_jobs': int((tardiness > 0).sum()),
        'due_date_adherence_rate': round(float((tardiness == 0).mean()), 4),
        'total_tardiness_min': round(float(tardiness.sum()), 2),
        'avg_tardiness_min': round(float(tardiness.mean()), 2),
        'makespan_start': str(schedule.start_time.min()),
        'makespan_end': str(schedule.end_time.max()),
        'total_setup_time_min': int(schedule.setup_time_min.sum()),
        'total_energy_kWh_est': round(float(schedule.energy_kWh_est.sum()), 3),
        'total_energy_cost_est': round(float(schedule.energy_cost_est.sum()), 2),
        'total_peak_overlap_slots': int(schedule.peak_overlap_slots.sum()),
        'total_co2_kg_est': round(float(schedule.co2_kg_est.sum()), 3),
        'policy_mode_counts': schedule['slack_policy_mode'].value_counts().to_dict()
    }


def load_csv(inp, original, data_id):
    return pd.read_csv(inp/f'data_{data_id}.csv') if (inp/f'data_{data_id}.csv').exists() else pd.read_csv(inp/original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='/home/user/input')
    ap.add_argument('--output-dir', default='/home/user/output')
    args = ap.parse_args()
    inp = Path(args.input_dir); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    job_orders = load_csv(inp, 'job_orders.csv', 7148714)
    machines = load_csv(inp, 'machines.csv', 7148716)
    maint = load_csv(inp, 'maintenance_schedule.csv', 7148718)
    routing = load_csv(inp, 'product_routing.csv', 7148720)
    tariff = load_csv(inp, 'tou_tariff.csv', 7148722)

    schedule = build_schedule(job_orders, machines, routing, maint, tariff)
    schedule.to_csv(out/'slack_aware_energy_schedule.csv', index=False, encoding='utf-8-sig')
    metrics = compute_metrics(schedule, job_orders)
    with open(out/'slack_aware_energy_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
