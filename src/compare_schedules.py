"""
Compare FIFO, EDD, Energy-aware v1, Slack-aware Energy, Slack-aware Energy v2 schedules using common metrics.

- data-dir에 energy_context.csv(uci_energy_integration.py 생성)가 있으면 UCI 결합 평가를 사용
  (6주 확장 TOU, UCI 배출계수, UCI 기저부하 기반 공장 수요전력, 설비 대기전력)
- 없으면 기존 tou_tariff.csv 기반으로 평가
"""
import argparse
from pathlib import Path

import pandas as pd
from metrics import calculate_metrics, save_metrics

SCHEDULE_FILES = {
    "FIFO": "fifo_schedule.csv",
    "EDD": "edd_schedule.csv",
    "Energy_Aware_v1": "energy_aware_schedule.csv",
    "Slack_Aware_Energy": "slack_aware_energy_schedule.csv",
    "Slack_Aware_Energy_v2": "slack_aware_energy_v2_schedule.csv",
}


def pct(a, b):
    return round((a - b) / b * 100, 2) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./virtual_data")
    ap.add_argument("--schedule-dir", default="./output")
    ap.add_argument("--output-dir", default="./output")
    args = ap.parse_args()
    data_dir, schedule_dir, out = Path(args.data_dir), Path(args.schedule_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def d(original, fid):
        p1 = data_dir / original
        p2 = data_dir / f"data_{fid}.csv"
        return pd.read_csv(p1 if p1.exists() else p2)

    job_orders = d("job_orders.csv", 7148714)
    machines = d("machines.csv", 7148716)
    maintenance = d("maintenance_schedule.csv", 7148718)
    tariff = d("tou_tariff.csv", 7148722)
    ctx_path = data_dir / "energy_context.csv"
    context = pd.read_csv(ctx_path) if ctx_path.exists() else None
    print("evaluation:", "UCI energy_context" if context is not None else "tou_tariff only")

    all_metrics = []
    for name, fname in SCHEDULE_FILES.items():
        path = schedule_dir / fname
        if not path.exists():
            print(f"SKIP: {fname} not found")
            continue
        sched = pd.read_csv(path)
        metrics, normalized, profile = calculate_metrics(
            sched, job_orders, machines, tariff, name, context=context, maintenance=maintenance)
        all_metrics.append(metrics)
        save_metrics(metrics, out / f"{name.lower()}_common_metrics.json")
        normalized.to_csv(out / f"{name.lower()}_schedule_common.csv", index=False, encoding="utf-8-sig")
        profile.to_csv(out / f"{name.lower()}_power_profile.csv", index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame([{k: v for k, v in m.items() if k != "machine_busy_hours"} for m in all_metrics])
    comparison.to_csv(out / "common_schedule_comparison.csv", index=False, encoding="utf-8-sig")

    if "EDD" in comparison["schedule_name"].values:
        edd = comparison[comparison.schedule_name == "EDD"].iloc[0]
        rows = []
        for _, r in comparison.iterrows():
            if r.schedule_name == "EDD":
                continue
            rows.append({
                "schedule_name": r.schedule_name,
                "due_rate_diff_pctp_vs_edd": round((r.due_date_adherence_rate - edd.due_date_adherence_rate) * 100, 2),
                "tardiness_change_pct_vs_edd": pct(r.total_tardiness_min, edd.total_tardiness_min),
                "energy_cost_change_pct_vs_edd": pct(r.total_energy_cost, edd.total_energy_cost),
                "peak_slots_change_pct_vs_edd": pct(r.total_peak_overlap_slots, edd.total_peak_overlap_slots),
                "peak_energy_change_pct_vs_edd": pct(r.peak_period_energy_kWh, edd.peak_period_energy_kWh),
                "facility_max_demand_change_pct_vs_edd": pct(r.facility_max_demand_kW, edd.facility_max_demand_kW),
                "energy_incl_idle_change_pct_vs_edd": pct(r.total_energy_incl_idle_kWh, edd.total_energy_incl_idle_kWh),
                "co2_incl_idle_change_pct_vs_edd": pct(r.total_co2_incl_idle_kg, edd.total_co2_incl_idle_kg),
                "setup_time_change_pct_vs_edd": pct(r.total_setup_time_min, edd.total_setup_time_min),
            })
        pd.DataFrame(rows).to_csv(out / "improvement_vs_edd.csv", index=False, encoding="utf-8-sig")

    show = ["schedule_name", "on_time_jobs", "due_date_adherence_rate", "total_tardiness_min",
            "total_setup_time_min", "total_energy_cost", "total_peak_overlap_slots", "peak_period_energy_kWh",
            "facility_max_demand_kW", "idle_energy_kWh", "total_co2_incl_idle_kg", "makespan_end",
            "ops_outside_energy_context"]
    print(comparison[[c for c in show if c in comparison]].to_string(index=False))


if __name__ == "__main__":
    main()
