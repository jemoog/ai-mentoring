"""
스케줄 제약조건 검증기.

검사 항목
- release_time 이전 시작 금지
- 제품별 공정 순서 (이전 공정 종료 후 다음 공정 시작)
- 적격 설비 배정
- 설비 중복(동시 2작업) 금지
- 설비 일일 가동시간 준수
- 정비시간 회피
- 셋업 정합성: 같은 설비에서 '시간상 직전' 공정의 제품이 다르면 셋업이 있어야 함
  (기존 스케줄러는 '마지막으로 배정한 제품' 기준이라 빈 구간 삽입 시 셋업 누락이 생길 수 있음)

실행: python src/validate_schedule.py --data-dir ./virtual_data --schedule-dir ./output_uci
"""
import argparse
import json
from pathlib import Path

import pandas as pd

FILES = {
    "FIFO": "fifo_schedule.csv",
    "EDD": "edd_schedule.csv",
    "Energy_Aware_v1": "energy_aware_schedule.csv",
    "Slack_Aware_Energy": "slack_aware_energy_schedule.csv",
    "Slack_Aware_Energy_v2": "slack_aware_energy_v2_schedule.csv",
}


def minutes_of_day(hhmm):
    return 24 * 60 if hhmm == "23:59" else int(hhmm[:2]) * 60 + int(hhmm[3:])


def validate(s, jobs, machines, routing, maint):
    s = s.copy()
    for c in ["start_time", "end_time"]:
        s[c] = pd.to_datetime(s[c])
    jobs = jobs.set_index("job_id")
    m = machines.set_index("machine_id")
    maint = maint.copy()
    for c in ["maintenance_start", "maintenance_end"]:
        maint[c] = pd.to_datetime(maint[c])
    elig = set(zip(routing.product_type, routing.process_step, routing.eligible_machine_id))
    v = {"release": 0, "precedence": 0, "eligibility": 0, "machine_overlap": 0,
         "availability": 0, "maintenance": 0, "missing_setup": 0, "missing_ops": 0}

    for jid, g in s.groupby("job_id"):
        g = g.sort_values("process_step")
        if g.start_time.min() < pd.Timestamp(jobs.loc[jid, "release_time"]):
            v["release"] += 1
        if len(g) != int(jobs.loc[jid, "required_process_count"]):
            v["missing_ops"] += 1
        ends = g.end_time.tolist(); starts = g.start_time.tolist()
        v["precedence"] += sum(starts[i + 1] < ends[i] for i in range(len(g) - 1))
    for _, r in s.iterrows():
        if (r.product_type, r.process_step, r.machine_id) not in elig:
            v["eligibility"] += 1
        row = m.loc[r.machine_id]
        a, b = minutes_of_day(row.available_start_time), minutes_of_day(row.available_end_time)
        if b - a < 24 * 60:  # 24시간 설비가 아니면 같은 날 가동시간 안에 있어야 함
            day = r.start_time.normalize()
            if not (r.start_time >= day + pd.Timedelta(minutes=a) and r.end_time <= day + pd.Timedelta(minutes=b)):
                v["availability"] += 1
        mm = maint[maint.machine_id == r.machine_id]
        if ((mm.maintenance_start < r.end_time) & (r.start_time < mm.maintenance_end)).any():
            v["maintenance"] += 1
    for mid, g in s.sort_values("start_time").groupby("machine_id"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            if g.start_time[i] < g.end_time[i - 1]:
                v["machine_overlap"] += 1
            if g.product_type[i] != g.product_type[i - 1] and int(g.setup_time_min[i]) == 0:
                v["missing_setup"] += 1
    v["total_violations"] = int(sum(v.values()))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./virtual_data")
    ap.add_argument("--schedule-dir", default="./output_uci")
    args = ap.parse_args()
    d, sd = Path(args.data_dir), Path(args.schedule_dir)
    jobs = pd.read_csv(d / "job_orders.csv")
    machines = pd.read_csv(d / "machines.csv")
    routing = pd.read_csv(d / "product_routing.csv")
    maint = pd.read_csv(d / "maintenance_schedule.csv")
    report = {}
    for name, f in FILES.items():
        if (sd / f).exists():
            report[name] = validate(pd.read_csv(sd / f), jobs, machines, routing, maint)
    out = pd.DataFrame(report).T
    print(out.to_string())
    out.to_csv(sd / "constraint_validation.csv", encoding="utf-8-sig")
    with open(sd / "constraint_validation.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
