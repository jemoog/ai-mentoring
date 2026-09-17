"""
Compare EDD, Energy-aware v1, and Slack-aware Energy schedules using common metrics.
"""
import argparse
from pathlib import Path
import pandas as pd
from metrics import calculate_metrics, save_metrics


def load_csv(base, name):
    return pd.read_csv(base / name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/home/user/input')
    ap.add_argument('--schedule-dir', default='/home/user/output')
    ap.add_argument('--output-dir', default='/home/user/output')
    args = ap.parse_args()
    data_dir = Path(args.data_dir); schedule_dir = Path(args.schedule_dir); out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Supports both local original names and FactChat downloaded names.
    def d(original, fid):
        p1 = data_dir / original
        p2 = data_dir / f'data_{fid}.csv'
        return pd.read_csv(p1 if p1.exists() else p2)

    job_orders = d('job_orders.csv', 7148714)
    machines = d('machines.csv', 7148716)
    tariff = d('tou_tariff.csv', 7148722)

    schedule_files = {
        'EDD': 'edd_schedule.csv',
        'Energy_Aware_v1': 'energy_aware_schedule.csv',
        'Slack_Aware_Energy': 'slack_aware_energy_schedule.csv',
    }

    all_metrics = []
    for name, fname in schedule_files.items():
        path = schedule_dir / fname
        if not path.exists():
            print(f'SKIP: {fname} not found')
            continue
        sched = pd.read_csv(path)
        metrics, normalized, profile = calculate_metrics(sched, job_orders, machines, tariff, name)
        all_metrics.append(metrics)
        save_metrics(metrics, out / f'{name.lower()}_common_metrics.json')
        normalized.to_csv(out / f'{name.lower()}_schedule_common.csv', index=False, encoding='utf-8-sig')
        profile.to_csv(out / f'{name.lower()}_power_profile.csv', index=False, encoding='utf-8-sig')

    comparison = pd.DataFrame([{k:v for k,v in m.items() if k != 'machine_busy_hours'} for m in all_metrics])
    comparison.to_csv(out / 'common_schedule_comparison.csv', index=False, encoding='utf-8-sig')

    # Improvement table vs EDD where applicable
    if 'EDD' in comparison['schedule_name'].values:
        edd = comparison[comparison.schedule_name == 'EDD'].iloc[0]
        rows = []
        for _, r in comparison.iterrows():
            if r.schedule_name == 'EDD':
                continue
            rows.append({
                'schedule_name': r.schedule_name,
                'due_rate_diff_pctp_vs_edd': round((r.due_date_adherence_rate - edd.due_date_adherence_rate) * 100, 2),
                'energy_cost_change_pct_vs_edd': round((r.total_energy_cost - edd.total_energy_cost) / edd.total_energy_cost * 100, 2),
                'peak_slots_change_pct_vs_edd': round((r.total_peak_overlap_slots - edd.total_peak_overlap_slots) / edd.total_peak_overlap_slots * 100, 2) if edd.total_peak_overlap_slots else None,
                'co2_change_pct_vs_edd': round((r.total_co2_kg - edd.total_co2_kg) / edd.total_co2_kg * 100, 2),
                'tardiness_change_pct_vs_edd': round((r.total_tardiness_min - edd.total_tardiness_min) / edd.total_tardiness_min * 100, 2) if edd.total_tardiness_min else None,
            })
        pd.DataFrame(rows).to_csv(out / 'improvement_vs_edd.csv', index=False, encoding='utf-8-sig')

    print(comparison.to_string(index=False))

if __name__ == '__main__':
    main()
