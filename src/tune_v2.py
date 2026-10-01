"""
Slack-aware v2 파라미터 튜닝 (2단계)

1단계: EDD 순서 고정(iterations=0)으로 배정 정책 파라미터 그리드 탐색 → 빠름
2단계: 1단계 상위 후보에 대해 작업 순서 local search 실행 → 최종 선택

선택 기준(납기 우선): 총지연(분) + 지연건수x600 + 에너지비용x0.02 (= v2 목적함수)
결과: output_uci/tuning_stage1.csv, tuning_stage2.csv, best_v2_params.json

실행: python src/tune_v2.py --input-dir ./virtual_data --output-dir ./output_uci --top 4 --iterations 400
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd

import slack_aware_energy_scheduler_v2 as v2

GRID = {
    "shift_load_types": [["Maximum"], ["Maximum", "Medium"]],
    "min_slack_h": [4.0, 8.0, 16.0],
    "slack_use_ratio": [0.3, 0.6],
    "max_window_h": [12.0, 24.0],
    "bottleneck_window_h": [0.0, 2.0, 6.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="./virtual_data")
    ap.add_argument("--output-dir", default="./output_uci")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=400)
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = v2.Problem(pd.read_csv(inp / "job_orders.csv"), pd.read_csv(inp / "machines.csv"),
                   pd.read_csv(inp / "product_routing.csv"), pd.read_csv(inp / "maintenance_schedule.csv"),
                   pd.read_csv(inp / "energy_context.csv"))
    order = v2.edd_order(P)

    rows = []
    keys = list(GRID)
    t0 = time.time()
    for combo in itertools.product(*GRID.values()):
        params = {**v2.DEFAULT_PARAMS, **dict(zip(keys, combo))}
        r = v2.simulate(P, order, params)
        rows.append({**{k: (",".join(v) if isinstance(v, list) else v) for k, v in zip(keys, combo)},
                     "objective": round(r["objective"]), "total_tardiness_min": r["total_tardiness_min"],
                     "tardy_jobs": r["tardy_jobs"], "energy_cost": round(r["energy_cost"]),
                     "peak_slots": r["peak_slots"]})
    s1 = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    s1.to_csv(out / "tuning_stage1.csv", index=False, encoding="utf-8-sig")
    print(f"stage1: {len(s1)} configs in {time.time() - t0:.0f}s")
    print(s1.head(10).to_string(index=False))

    rows2, best = [], None
    for _, r in s1.head(args.top).iterrows():
        params = {**v2.DEFAULT_PARAMS, **{k: (r[k].split(",") if k == "shift_load_types" else float(r[k])) for k in keys},
                  "iterations": args.iterations}
        _, res = v2.local_search(P, params, log=lambda *a: None)
        row = {**{k: r[k] for k in keys}, "objective": round(res["objective"]),
               "total_tardiness_min": res["total_tardiness_min"], "tardy_jobs": res["tardy_jobs"],
               "energy_cost": round(res["energy_cost"]), "peak_slots": res["peak_slots"]}
        rows2.append(row)
        print("stage2:", row)
        if best is None or res["objective"] < best[0]:
            best = (res["objective"], params)
    s2 = pd.DataFrame(rows2).sort_values("objective")
    s2.to_csv(out / "tuning_stage2.csv", index=False, encoding="utf-8-sig")
    best_params = {k: best[1][k] for k in keys}
    with open(out / "best_v2_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print("best params:", best_params)


if __name__ == "__main__":
    main()
