"""
UCI 실측 주간 테스트.

기본 평가는 UCI 1년치를 '요일 x 15분' 평균으로 만든 기저부하 프로파일을 쓴다.
평균은 실제 변동(특정 날의 급등, 주별 생산량 차이)을 지운다. 이 스크립트는
2018년 실제 연속 6주 구간(월요일 시작, 52개)을 그대로 기저부하로 넣어,
스케줄별 공장 최대수요전력(라인 + 실측 기저부하)이 실측 변동에서도 유지되는지 확인한다.

입력: experiments/runs/<scenario>/ 의 스케줄 (run_experiments.py 결과)
출력: experiments/uci_real_weeks.csv (주 x 시나리오 x 방식), experiments/uci_real_weeks_summary.csv

실행: python src/uci_real_week_test.py --scenarios load070 load085 load100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import SlotGrid, _op_power_series, _clip, _to_dt
from uci_energy_integration import load_uci

SLOT_MIN = 15
METHODS = {"EDD": "edd_schedule.csv", "EDD_setup_correct": "edd_setup_correct_schedule.csv",
           "Energy_Aware_v1": "energy_aware_schedule.csv", "Slack_Aware_v2": "slack_aware_v2_schedule.csv"}


def line_profile(sched, machines, ctx):
    grid = SlotGrid(ctx)
    sched = _to_dt(sched, ["start_time", "end_time"]).reset_index(drop=True)
    line = np.zeros(grid.n)
    for _i, _m, a, b, p, w in _op_power_series(sched, machines, grid):
        a, b, w = _clip(a, b, w, grid.n)
        line[a:b] += p * w
    a0, b0 = grid.span(sched.start_time.min(), sched.end_time.max())
    return line, grid.peak.astype(bool), max(a0, 0), min(b0, grid.n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci-path", default="./uci_data/Steel_industry_data.csv")
    ap.add_argument("--exp-dir", default="./experiments")
    ap.add_argument("--scenarios", nargs="+", default=["load070", "load085", "load100"])
    args = ap.parse_args()
    exp = Path(args.exp_dir)

    uci = load_uci(args.uci_path)
    kw = uci.set_index("slot_start")["kW"]
    mondays = [d for d in pd.date_range("2018-01-01", "2018-12-31", freq="W-MON")]

    rows = []
    for sdir in sorted((exp / "runs").iterdir()):
        if not any(sdir.name.startswith(p) for p in args.scenarios):
            continue
        data = exp / "data" / sdir.name
        ctx = pd.read_csv(data / "energy_context.csv")
        machines = pd.read_csv(data / "machines.csv")
        n = len(ctx)
        profiles = {}
        for name, f in METHODS.items():
            if (sdir / f).exists():
                profiles[name] = line_profile(pd.read_csv(sdir / f), machines, ctx)
        for mon in mondays:
            idx = pd.date_range(mon, periods=n, freq=f"{SLOT_MIN}min")
            # 연말을 넘으면 연초로 순환 (같은 요일 정렬 유지를 위해 364일 단위)
            idx = idx.where(idx.year == 2018, idx - pd.Timedelta(days=364))
            base = kw.reindex(idx).interpolate(limit_direction="both").to_numpy()
            for name, (line, peak, a, b) in profiles.items():
                fac = (line + base)[a:b]
                pk = peak[a:b]
                rows.append({"scenario": sdir.name, "week_start": mon.date(), "method": name,
                             "facility_max_demand_kW": round(float(fac.max()), 1),
                             "facility_peak_time_max_kW": round(float(fac[pk].max()), 1),
                             "base_max_kW": round(float(base[a:b].max()), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(exp / "uci_real_weeks.csv", index=False, encoding="utf-8-sig")

    # EDD 대비: 같은 시나리오·같은 주에서 비교
    piv = df.pivot_table(index=["scenario", "week_start"], columns="method",
                         values=["facility_max_demand_kW", "facility_peak_time_max_kW"])
    out = []
    for col in ["facility_max_demand_kW", "facility_peak_time_max_kW"]:
        for m in [c for c in piv[col].columns if c != "EDD"]:
            diff = piv[col][m] - piv[col]["EDD"]
            g = diff.groupby(level="scenario")
            load = diff.index.get_level_values("scenario").str[:7]
            for ld, dd in diff.groupby(load):
                out.append({"metric": col, "method": m, "load": ld, "weeks_x_seeds": int(dd.size),
                            "mean_diff_kW": round(float(dd.mean()), 1),
                            "mean_diff_pct": round(float((dd / piv[col]["EDD"].loc[dd.index]).mean() * 100), 2),
                            "share_lower_than_EDD": round(float((dd < 0).mean()), 3),
                            "share_higher_than_EDD": round(float((dd > 0).mean()), 3)})
    summ = pd.DataFrame(out)
    summ.to_csv(exp / "uci_real_weeks_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
