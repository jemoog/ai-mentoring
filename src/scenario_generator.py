"""
가상 작업지시 시나리오 생성기 (설비 부하율 조절).

원본 가상 데이터는 병목 설비 부하율이 약 148%(1주 가용시간 대비 필요시간)라
어떤 알고리즘도 납기 대부분을 맞출 수 없고, 에너지 최적화(시간 이동) 여지도 거의 없다.
이 모듈은 작업 수량을 조절해 '가장 바쁜 설비의 부하율'을 목표값에 맞춘 시나리오를 만든다.

- seed 0 : 원본 job_orders.csv의 작업(제품·투입·납기·우선순위)을 그대로 두고 수량만 조정
           → 부하율 효과만 분리해서 볼 수 있음
- seed 1+: 원본과 같은 분포로 작업을 새로 생성한 뒤 수량 조정 (결과의 견고성 확인용)
           제품 비율 B 44% / A 34% / C 22%, 투입 9/21~9/23 (15분 단위),
           납기 = 투입 + 25~128시간 (계획 종료 9/28 00:00 이전), 우선순위 1~3 균등, 수량 56~298

부하율 정의 (Slack-aware v2의 load_ratio와 동일)
  설비별 필요시간 = Σ ceil(max(기준 가공시간, 수량/시간당능력×60) / 15분) × 15분
  설비별 가용시간 = 계획 1주 중 일일 가동시간 − 정비시간
  부하율 = 필요시간 / 가용시간,  목표 부하율 = max(설비별 부하율)

실행 예:
python src/scenario_generator.py --data-dir ./virtual_data --out-dir ./experiments/data --loads 0.7 0.85 1.0 1.2 1.48 --seeds 0 1 2
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SLOT_MIN = 15
PLAN_START = pd.Timestamp("2026-09-21 00:00:00")
PLAN_END = pd.Timestamp("2026-09-28 00:00:00")
SHARED_FILES = ["machines.csv", "product_routing.csv", "maintenance_schedule.csv",
                "tou_tariff.csv", "tou_tariff_extended.csv", "energy_context.csv"]


def available_hours(machines, maint):
    """설비별 계획 1주 가용시간 (일일 가동시간 − 정비)."""
    slots = pd.date_range(PLAN_START, PLAN_END, freq=f"{SLOT_MIN}min", inclusive="left")
    tod = slots.hour * 60 + slots.minute
    out = {}
    for _, m in machines.iterrows():
        s = int(m.available_start_time[:2]) * 60 + int(m.available_start_time[3:])
        e = 24 * 60 if m.available_end_time == "23:59" else int(m.available_end_time[:2]) * 60 + int(m.available_end_time[3:])
        ok = (tod >= s) & (tod < e)
        for _, r in maint[maint.machine_id == m.machine_id].iterrows():
            ok &= ~((slots >= pd.Timestamp(r.maintenance_start)) & (slots < pd.Timestamp(r.maintenance_end)))
        out[m.machine_id] = ok.sum() * SLOT_MIN / 60
    return out


def load_ratios(jobs, machines, routing, avail):
    cap = machines.set_index("machine_id")["capacity_per_hour"].to_dict()
    need = {m: 0.0 for m in avail}
    for _, j in jobs.iterrows():
        for _, rt in routing[routing.product_type == j.product_type].iterrows():
            m = rt.eligible_machine_id
            proc = max(rt.process_time_min, j.quantity / cap[m] * 60)
            need[m] += math.ceil(proc / SLOT_MIN) * SLOT_MIN / 60
    return {m: need[m] / avail[m] for m in avail}


def scale_to_load(jobs, target, machines, routing, avail, q_min=10):
    """수량 배율을 이분탐색해 max 부하율을 target에 맞춘다."""
    base_q = jobs["quantity"].astype(float).to_numpy()

    def ratio_at(f):
        j = jobs.copy()
        j["quantity"] = np.maximum(q_min, np.round(base_q * f)).astype(int)
        return max(load_ratios(j, machines, routing, avail).values()), j

    lo, hi = 0.05, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        r, _ = ratio_at(mid)
        if r < target:
            lo = mid
        else:
            hi = mid
    r_lo, j_lo = ratio_at(lo)
    r_hi, j_hi = ratio_at(hi)
    return (j_lo, lo, r_lo) if abs(r_lo - target) <= abs(r_hi - target) else (j_hi, hi, r_hi)


def generate_jobs(seed, n=80):
    rng = np.random.default_rng(seed)
    products = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.44, 0.22])
    rel_slots = rng.integers(0, 3 * 24 * 4, size=n)  # 9/21 00:00 ~ 9/23 23:45
    release = PLAN_START + pd.to_timedelta(rel_slots * SLOT_MIN, unit="min")
    window_h = rng.integers(25, 129, size=n)
    due = release + pd.to_timedelta(window_h, unit="h")
    due = due.where(due <= PLAN_END - pd.Timedelta(hours=2), PLAN_END - pd.Timedelta(hours=2))
    jobs = pd.DataFrame({
        "job_id": [f"J{i + 1:03d}" for i in range(n)],
        "product_type": products,
        "quantity": rng.integers(56, 299, size=n),
        "release_time": release,
        "due_time": due,
        "priority": rng.integers(1, 4, size=n),
        "required_process_count": 3,
    })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./virtual_data")
    ap.add_argument("--out-dir", default="./experiments/data")
    ap.add_argument("--loads", type=float, nargs="+", default=[0.70, 0.85, 1.00, 1.20, 1.48])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.out_dir)
    machines = pd.read_csv(data / "machines.csv")
    routing = pd.read_csv(data / "product_routing.csv")
    maint = pd.read_csv(data / "maintenance_schedule.csv")
    avail = available_hours(machines, maint)
    original = pd.read_csv(data / "job_orders.csv")

    summary = []
    for seed in args.seeds:
        base = original.copy() if seed == 0 else generate_jobs(seed)
        base_ratio = load_ratios(base, machines, routing, avail)
        for target in args.loads:
            name = f"load{int(round(target * 100)):03d}_seed{seed}"
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            if seed == 0 and abs(target - max(base_ratio.values())) < 0.01:
                jobs, factor, ratio = base.copy(), 1.0, max(base_ratio.values())  # 원본 그대로
            else:
                jobs, factor, ratio = scale_to_load(base, target, machines, routing, avail)
            jobs = jobs.copy()
            for c in ["release_time", "due_time"]:
                jobs[c] = pd.to_datetime(jobs[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
            jobs.to_csv(d / "job_orders.csv", index=False, encoding="utf-8-sig")
            for f in SHARED_FILES:
                if (data / f).exists():
                    shutil.copy(data / f, d / f)
            per_m = load_ratios(jobs, machines, routing, avail)
            row = {"scenario": name, "target_load": target, "seed": seed, "quantity_factor": round(factor, 4),
                   "max_load_ratio": round(ratio, 3), "total_quantity": int(jobs.quantity.sum()),
                   **{f"load_{m}": round(r, 3) for m, r in per_m.items()}}
            summary.append(row)
            print(row)
    pd.DataFrame(summary).to_csv(out / "scenarios.csv", index=False, encoding="utf-8-sig")
    with open(out / "scenarios.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
