"""
대체 설비 시나리오 생성.

scenario_generator.py가 만든 시나리오(experiments/data/<scenario>)와 '같은 작업지시'를 쓰되,
설비·라우팅·정비 데이터만 virtual_data/alt_machines/ 의 대체 설비 버전으로 바꾼다.

대체 설비 (기존 설비보다 느리고 전력 효율이 낮게 가정)
- M06 Heat Treatment Furnace 2 (old): B·C 열처리, 32개/h, 165kW  → 단위 에너지 M03 대비 +43%
- M07 Assembly Cell 2 (manual)     : B 조립·C 조립, 45개/h, 48kW → 단위 에너지 M02 대비 +16%

→ 같은 수요에서 설비 능력만 늘어난 상황. 대체 설비를 쓰면 납기는 좋아지지만 에너지는 늘어나는
  trade-off를 스케줄러가 어떻게 다루는지 본다.

실행: python src/make_alt_scenarios.py --base-dir ./experiments/data --alt-src ./virtual_data/alt_machines --out-dir ./experiments/data_alt
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slack_aware_energy_scheduler_v2 as v2  # noqa: E402

ALT_FILES = ["machines.csv", "product_routing.csv", "maintenance_schedule.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="./experiments/data")
    ap.add_argument("--alt-src", default="./virtual_data/alt_machines")
    ap.add_argument("--out-dir", default="./experiments/data_alt")
    args = ap.parse_args()
    base, alt, out = Path(args.base_dir), Path(args.alt_src), Path(args.out_dir)
    scen = pd.read_csv(base / "scenarios.csv")
    rows = []
    for s in scen.scenario:
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        for f in (base / s).iterdir():
            if f.is_file():
                shutil.copy(f, d / f.name)
        for f in ALT_FILES:
            shutil.copy(alt / f, d / f)
        P = v2.Problem(*[pd.read_csv(d / f) for f in
                         ["job_orders.csv", "machines.csv", "product_routing.csv", "maintenance_schedule.csv",
                          "energy_context.csv"]])
        row = scen[scen.scenario == s].iloc[0].to_dict()
        row["pooled_max_load_ratio"] = round(max(P.load_ratio.values()), 3)
        row.update({f"pooled_load_{m}": round(r, 3) for m, r in P.load_ratio.items()})
        rows.append(row)
        print(s, "pooled max load", row["pooled_max_load_ratio"])
    pd.DataFrame(rows).to_csv(out / "scenarios.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
