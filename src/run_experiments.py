"""
시나리오 실험 일괄 실행 + 결과 집계.

1) scenario_generator.py가 만든 시나리오(부하율 x seed)마다
   FIFO, EDD, Energy-aware v1, Slack-aware v1, Slack-aware v2 를 실행하고
   UCI 결합 공통 지표(metrics.py)와 제약 검증(validate_schedule.py)으로 평가
2) Ablation (기본: 부하율 85% 시나리오): v2의 구성요소를 하나씩 끈 변형을 비교
   - v2_no_search      : 작업 순서 탐색 없음 (EDD 순서 + v2 배정 규칙)
   - v2_shift_all      : Light/Medium 공정까지 시간 이동 허용
   - v2_no_bottleneck  : 병목 설비 보호 끔
   - v2_energy_heavy   : 목표함수의 전력비 가중 10배 (0.02 → 0.2)
3) 결과
   experiments/results_long.csv      : 시나리오 x 방식별 전체 지표
   experiments/summary_by_load.csv   : 부하율별 평균 (seed 평균)
   experiments/vs_edd_by_load.csv    : 부하율별 EDD 대비 변화율 (seed 평균)
   experiments/ablation.csv          : ablation 결과

실행 (프로젝트 루트에서, 약 15~25분):
python src/scenario_generator.py
python src/run_experiments.py --workers 2 --iterations 1000
"""
import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import slack_aware_energy_scheduler_v2 as v2  # noqa: E402
from metrics import calculate_metrics  # noqa: E402
from validate_schedule import validate  # noqa: E402

BASELINES = [("FIFO", "fifo_scheduler.py", "fifo_schedule.csv"),
             ("EDD", "edd_scheduler.py", "edd_schedule.csv"),
             ("Energy_Aware_v1", "energy_aware_scheduler.py", "energy_aware_schedule.csv"),
             ("Slack_Aware_v1", "slack_aware_energy_scheduler.py", "slack_aware_energy_schedule.csv")]

ABLATIONS = {
    "v2_no_search": {"iterations": 0},
    "v2_shift_all": {"shift_load_types": ["Light", "Medium", "Maximum"]},
    "v2_no_bottleneck": {"bottleneck_threshold": 99.0},
    "v2_short_slack": {"min_slack_h": 8.0},
    "v2_energy_heavy": {"obj_cost_per_min": 0.2},
}

KEEP = ["on_time_jobs", "due_date_adherence_rate", "total_tardiness_min", "avg_tardiness_min", "max_tardiness_min",
        "makespan_end", "total_setup_time_min", "total_energy_kWh", "idle_energy_kWh", "total_energy_incl_idle_kWh",
        "total_energy_cost", "avg_price_per_kWh", "total_peak_overlap_slots", "peak_period_energy_kWh",
        "peak_period_energy_share", "max_demand_kW", "facility_max_demand_kW", "facility_peak_time_max_demand_kW",
        "demand_charge_est_month",
        "total_co2_incl_idle_kg"]


def evaluate(name, sched, d, out):
    jobs = pd.read_csv(d / "job_orders.csv")
    machines = pd.read_csv(d / "machines.csv")
    maint = pd.read_csv(d / "maintenance_schedule.csv")
    routing = pd.read_csv(d / "product_routing.csv")
    ctx = pd.read_csv(d / "energy_context.csv")
    m, _, _ = calculate_metrics(sched, jobs, machines, None, name, context=ctx, maintenance=maint)
    v = validate(sched, jobs, machines, routing, maint)
    # 대체 설비 사용 공정 수: (제품, 단계)별 설비 ID가 가장 작은 설비를 기본 설비로 보고 나머지를 대체로 집계
    primary = routing.groupby(["product_type", "process_step"]).eligible_machine_id.min().to_dict()
    alt_ops = int(sum(primary[(r.product_type, r.process_step)] != r.machine_id for r in sched.itertuples()))
    row = {"method": name, **{k: m[k] for k in KEEP}, "violations": v["total_violations"],
           "missing_setup": v["missing_setup"], "alt_machine_ops": alt_ops}
    return row


def run_scenario(task):
    d, out, iterations, ablate = Path(task["data"]), Path(task["out"]), task["iterations"], task["ablate"]
    extra = task.get("extra", {})
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for name, script, fname in BASELINES:
        if task.get("skip_baselines") and (out / fname).exists():
            rows.append(evaluate(name, pd.read_csv(out / fname), d, out))
            continue
        r = subprocess.run([sys.executable, str(SRC / script), "--input-dir", str(d), "--output-dir", str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{d.name} {name}: {r.stderr[-500:]}")
        rows.append(evaluate(name, pd.read_csv(out / fname), d, out))

    args = (pd.read_csv(d / "job_orders.csv"), pd.read_csv(d / "machines.csv"), pd.read_csv(d / "product_routing.csv"),
            pd.read_csv(d / "maintenance_schedule.csv"), pd.read_csv(d / "energy_context.csv"))
    # EDD_setup_correct: EDD 순서 + v2 배정 규칙(셋업 정합성 보장), 시간 이동·순서 탐색 없음 → 공정한 납기 기준선
    variants = {"EDD_setup_correct": {"iterations": 0, "shift_load_types": []},
                "Slack_Aware_v2": {"iterations": iterations}}
    variants.update({k: {"iterations": iterations, **p} for k, p in extra.items()})
    if ablate:
        variants.update({k: {"iterations": iterations, **p} for k, p in ABLATIONS.items()})
    for name, params in variants.items():
        sched, _, _, _ = v2.build_schedule(*[a.copy() for a in args], params=params, log=lambda *a: None)
        sched.to_csv(out / f"{name.lower()}_schedule.csv", index=False, encoding="utf-8-sig")
        row = evaluate(name, sched, d, out)
        row["shifted_operations"] = int((sched.shift_min > 0).sum())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "scenario", d.name)
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    print(f"[done] {d.name} ({time.time() - t0:.0f}s)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", default="./experiments/data")
    ap.add_argument("--out-dir", default="./experiments")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ablation-load", default="load085", help="ablation을 실행할 시나리오 접두어 (none이면 생략)")
    ap.add_argument("--extra-variants", default=None,
                    help='모든 시나리오에 추가로 돌릴 v2 변형 JSON 파일 {"이름": {파라미터}}')
    ap.add_argument("--skip-baselines", action="store_true", help="FIFO/EDD/v1 계열은 기존 결과 재사용")
    args = ap.parse_args()
    sdir, out = Path(args.scenario_dir), Path(args.out_dir)
    scen = pd.read_csv(sdir / "scenarios.csv")
    extra = json.load(open(args.extra_variants, encoding="utf-8")) if args.extra_variants else {}
    tasks = [{"data": str(sdir / s), "out": str(out / "runs" / s), "iterations": args.iterations,
              "ablate": s.startswith(args.ablation_load),
              "extra": extra, "skip_baselines": args.skip_baselines} for s in scen.scenario]
    # 오래 걸리는 시나리오(ablation, 고부하)부터 실행
    tasks.sort(key=lambda t: (not t["ablate"], t["data"]), reverse=False)
    with Pool(args.workers) as pool:
        dfs = pool.map(run_scenario, tasks, chunksize=1)

    res = pd.concat(dfs, ignore_index=True).merge(
        scen[["scenario", "target_load", "seed", "max_load_ratio"]], on="scenario")
    res.to_csv(out / "results_long.csv", index=False, encoding="utf-8-sig")
    aggregate(res, out)


def aggregate(res, out):
    main_methods = ["FIFO", "EDD", "EDD_setup_correct", "Energy_Aware_v1", "Slack_Aware_v1", "Slack_Aware_v2"]
    main_methods += [m for m in res.method.unique() if m.startswith("Slack_Aware_v2_")]
    num = ["on_time_jobs", "due_date_adherence_rate", "total_tardiness_min", "total_setup_time_min", "alt_machine_ops",
           "demand_charge_est_month",
           "total_energy_cost", "avg_price_per_kWh", "total_peak_overlap_slots", "peak_period_energy_kWh",
           "peak_period_energy_share", "facility_max_demand_kW", "total_energy_incl_idle_kWh",
           "total_co2_incl_idle_kg", "violations"]
    m = res[res.method.isin(main_methods)]
    summary = m.groupby(["target_load", "method"])[num].mean().round(3).reset_index()
    summary["method"] = pd.Categorical(summary.method, main_methods)
    summary.sort_values(["target_load", "method"]).to_csv(out / "summary_by_load.csv", index=False, encoding="utf-8-sig")

    # EDD 대비 변화율: 시나리오별로 계산 후 seed 평균
    edd = res[res.method == "EDD"].set_index("scenario")
    rows = []
    for _, r in res[res.method != "EDD"].iterrows():
        e = edd.loc[r.scenario]

        def pct(k):
            return (r[k] - e[k]) / e[k] * 100 if e[k] else 0.0

        rows.append({"scenario": r.scenario, "target_load": r.target_load, "seed": r.seed, "method": r.method,
                     "on_time_diff_pctp": (r.due_date_adherence_rate - e.due_date_adherence_rate) * 100,
                     "tardiness_min_diff": r.total_tardiness_min - e.total_tardiness_min,
                     "tardiness_pct": pct("total_tardiness_min"),
                     "energy_cost_pct": pct("total_energy_cost"),
                     "peak_slots_pct": pct("total_peak_overlap_slots"),
                     "peak_energy_pct": pct("peak_period_energy_kWh"),
                     "facility_max_demand_pct": pct("facility_max_demand_kW"),
                     "energy_incl_idle_pct": pct("total_energy_incl_idle_kWh"),
                     "co2_pct": pct("total_co2_incl_idle_kg"),
                     "setup_pct": pct("total_setup_time_min")})
    vs = pd.DataFrame(rows)
    vs.to_csv(out / "vs_edd_by_scenario.csv", index=False, encoding="utf-8-sig")
    cols = [c for c in vs.columns if c not in ("scenario", "target_load", "seed", "method")]
    g = vs.groupby(["target_load", "method"])[cols]
    agg = g.mean().round(2)
    # seed 간 일관성: EDD보다 전력비가 낮은 seed 수 / 총지연이 같거나 적은 seed 수
    agg["seeds"] = g.size()
    agg["seeds_cost_better"] = g.apply(lambda x: int((x.energy_cost_pct < 0).sum()))
    agg["seeds_tardiness_not_worse"] = g.apply(lambda x: int((x.tardiness_min_diff <= 0).sum()))
    agg.reset_index().to_csv(out / "vs_edd_by_load.csv", index=False, encoding="utf-8-sig")

    abl = res[res.method.str.startswith("v2_") | (res.method == "Slack_Aware_v2") | (res.method == "EDD")]
    abl = abl[abl.scenario.isin(res[res.method.str.startswith("v2_")].scenario.unique())]
    if len(abl):
        abl.groupby("method")[num + ["shifted_operations"]].mean().round(2).to_csv(
            out / "ablation.csv", encoding="utf-8-sig")
    print(agg.reset_index()[agg.reset_index().method == "Slack_Aware_v2"].to_string(index=False))


if __name__ == "__main__":
    main()
