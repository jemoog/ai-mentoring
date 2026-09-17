"""
EDD(Earliest Due Date) production scheduler for the virtual manufacturing dataset.
- Sort jobs by due_time, priority, release_time
- Respect process order, machine eligibility, daily availability, maintenance, and one-job-per-machine
- Add setup time when previous product on the same machine differs
- Export edd_schedule.csv and edd_metrics.json
"""
import argparse, json, math
from pathlib import Path
import pandas as pd

SLOT_MIN = 15

def ceil_to_slot(ts, slot_min=SLOT_MIN):
    ts = pd.Timestamp(ts)
    discard = pd.Timedelta(minutes=ts.minute % slot_min, seconds=ts.second, microseconds=ts.microsecond)
    floored = ts - discard
    return floored if floored == ts else floored + pd.Timedelta(minutes=slot_min)

def parse_hhmm(day, hhmm):
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
        # interval segment inside this day
        seg_end = min(end, day_start + pd.Timedelta(days=1))
        if not (start >= avs if cur == start else cur >= avs):
            return False
        if seg_end > ave:
            return False
        cur = day_start + pd.Timedelta(days=1)
    return True

def earliest_feasible(machine_id, product, duration_min, earliest_start, machines, machine_busy, maintenance, last_product):
    mrow = machines.loc[machines.machine_id == machine_id].iloc[0]
    setup_min = 0
    if last_product.get(machine_id) is not None and last_product[machine_id] != product:
        # setup time will be supplied by caller via duration total, here just informational fallback
        pass
    t = ceil_to_slot(earliest_start)
    horizon_end = pd.Timestamp(machines.attrs['horizon_end'])
    while t < horizon_end:
        end = t + pd.Timedelta(minutes=duration_min)
        if is_within_daily_availability(t, end, mrow):
            intervals = machine_busy.get(machine_id, []) + maintenance.get(machine_id, [])
            if not interval_conflict(t, end, intervals):
                return t, end
        t += pd.Timedelta(minutes=SLOT_MIN)
    raise RuntimeError(f'No feasible slot found for {machine_id} after {earliest_start}')

def compute_duration_min(quantity, route_row, machine_row):
    # Conservative lot duration: base process time + quantity/capacity time.
    capacity_time = quantity / machine_row['capacity_per_hour'] * 60
    return int(math.ceil(max(route_row['process_time_min'], capacity_time) / SLOT_MIN) * SLOT_MIN)

def build_edd_schedule(job_orders, machines, routing, maint):
    for col in ['release_time','due_time']:
        job_orders[col] = pd.to_datetime(job_orders[col])
    maint['maintenance_start'] = pd.to_datetime(maint['maintenance_start'])
    maint['maintenance_end'] = pd.to_datetime(maint['maintenance_end'])

    horizon_start = job_orders.release_time.min().floor('D')
    horizon_end = max(job_orders.due_time.max(), maint.maintenance_end.max()) + pd.Timedelta(days=30)
    machines.attrs['horizon_end'] = str(horizon_end)

    machine_busy = {m: [] for m in machines.machine_id}
    maintenance = {m: [] for m in machines.machine_id}
    for _, r in maint.iterrows():
        maintenance[r.machine_id].append((r.maintenance_start, r.maintenance_end))

    last_product = {m: None for m in machines.machine_id}
    schedule_rows = []

    jobs = job_orders.sort_values(['due_time','priority','release_time','job_id']).reset_index(drop=True)
    for _, job in jobs.iterrows():
        current_ready = job.release_time
        product_routes = routing[routing.product_type == job.product_type].sort_values('process_step')
        for _, route in product_routes.iterrows():
            candidates = []
            eligible = routing[(routing.product_type == job.product_type) & (routing.process_step == route.process_step)]
            for _, eroute in eligible.iterrows():
                mid = eroute.eligible_machine_id
                mrow = machines.loc[machines.machine_id == mid].iloc[0]
                setup_min = int(eroute.setup_time_min) if last_product[mid] is not None and last_product[mid] != job.product_type else 0
                proc_min = compute_duration_min(job.quantity, eroute, mrow)
                total_min = setup_min + proc_min
                try:
                    start, end = earliest_feasible(mid, job.product_type, total_min, current_ready, machines, machine_busy, maintenance, last_product)
                    candidates.append((end, start, mid, eroute, setup_min, proc_min, total_min))
                except RuntimeError:
                    continue
            if not candidates:
                raise RuntimeError(f'No feasible machine for job={job.job_id}, product={job.product_type}, step={route.process_step}')
            # EDD baseline: choose candidate with earliest completion time
            end, start, mid, chosen_route, setup_min, proc_min, total_min = sorted(candidates, key=lambda x: (x[0], x[1], x[2]))[0]
            machine_busy[mid].append((start, end))
            machine_busy[mid].sort()
            last_product[mid] = job.product_type
            current_ready = end
            mrow = machines.loc[machines.machine_id == mid].iloc[0]
            energy_kwh = round(mrow.operation_power_kW * (proc_min/60) + mrow.operation_power_kW * 0.2 * (setup_min/60), 3)
            schedule_rows.append({
                'job_id': job.job_id,
                'product_type': job.product_type,
                'quantity': int(job.quantity),
                'priority': int(job.priority),
                'due_time': job.due_time,
                'process_step': int(chosen_route.process_step),
                'process_id': chosen_route.process_id,
                'machine_id': mid,
                'load_type': chosen_route.load_type,
                'start_time': start,
                'end_time': end,
                'setup_time_min': setup_min,
                'processing_time_min': proc_min,
                'total_time_min': total_min,
                'operation_energy_kWh_est': energy_kwh
            })
    sched = pd.DataFrame(schedule_rows)
    return sched

def compute_metrics(schedule, job_orders):
    due = job_orders.set_index('job_id')['due_time'].apply(pd.Timestamp).to_dict()
    completion = schedule.groupby('job_id')['end_time'].max().apply(pd.Timestamp)
    tardiness_min = completion.index.to_series().map(lambda j: max(0, (completion[j]-due[j]).total_seconds()/60))
    return {
        'scheduled_operations': int(len(schedule)),
        'scheduled_jobs': int(completion.shape[0]),
        'on_time_jobs': int((tardiness_min == 0).sum()),
        'tardy_jobs': int((tardiness_min > 0).sum()),
        'due_date_adherence_rate': round(float((tardiness_min == 0).mean()), 4),
        'total_tardiness_min': round(float(tardiness_min.sum()), 2),
        'avg_tardiness_min': round(float(tardiness_min.mean()), 2),
        'makespan_start': str(schedule.start_time.min()),
        'makespan_end': str(schedule.end_time.max()),
        'total_setup_time_min': int(schedule.setup_time_min.sum()),
        'total_operation_energy_kWh_est': round(float(schedule.operation_energy_kWh_est.sum()), 3)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='/home/user/input')
    ap.add_argument('--output-dir', default='/home/user/output')
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # In local use, rename files to the original names, or keep these defaults.
    job_orders = pd.read_csv(inp/'data_7148714.csv') if (inp/'data_7148714.csv').exists() else pd.read_csv(inp/'job_orders.csv')
    machines = pd.read_csv(inp/'data_7148716.csv') if (inp/'data_7148716.csv').exists() else pd.read_csv(inp/'machines.csv')
    maint = pd.read_csv(inp/'data_7148718.csv') if (inp/'data_7148718.csv').exists() else pd.read_csv(inp/'maintenance_schedule.csv')
    routing = pd.read_csv(inp/'data_7148720.csv') if (inp/'data_7148720.csv').exists() else pd.read_csv(inp/'product_routing.csv')
    schedule = build_edd_schedule(job_orders, machines, routing, maint)
    schedule.to_csv(out/'edd_schedule.csv', index=False, encoding='utf-8-sig')
    metrics = compute_metrics(schedule, job_orders.assign(due_time=pd.to_datetime(job_orders.due_time)))
    with open(out/'edd_metrics.json','w',encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
