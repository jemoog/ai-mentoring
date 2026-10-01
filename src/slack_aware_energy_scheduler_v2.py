"""
Slack-aware Energy Scheduler v2 (UCI 결합 + 납기 보호 강화 + 순서 탐색)

v1(Slack-aware)의 한계
- slack을 '현재 공정'만 보고 계산 → 후속 공정 시간을 무시해 납기 위험을 과소평가
- 과부하(병목) 설비에서도 작업을 저요금 시간대로 미룸 → 설비 유휴 발생 → 뒤 작업 연쇄 지연
- Light/Medium 공정까지 이동 → 절감 효과는 작고 지연만 증가
- EDD 순서 고정 → 셋업/병목을 고려한 순서 개선 불가
- 1주 요금표 밖은 고정단가·피크0으로 계산 (확장 요금표 사용으로 해결)
- 셋업 판단이 '마지막으로 배정된 제품' 기준 → 빈 구간(gap) 삽입 시 실제 앞뒤 제품과 불일치

v2 핵심 아이디어
1) 하류공정 인지 slack: slack = 납기 - (최조 완료시각 + 후속 공정 최소 가공시간)
2) 병목 보호: 설비 부하율(필요시간/가용시간)이 임계값 이상이면 이동 범위를 크게 제한
3) 이동 대상 제한: 기본은 Maximum(고에너지) 공정만 시간 이동, 나머지는 최조 배정(EDD식)
4) 이동 범위 = min(최대창, slack x 비율) → 이동해도 납기 위반이 생기지 않는 범위 안에서만 탐색
5) UCI 기저부하 + 라인부하로 공장 수요전력을 계산하고 상한(demand cap) 초과분에 패널티
6) 셋업은 시간상 직전 공정의 제품 기준으로 판단, gap 삽입 시 직후 공정의 셋업 필요 여부까지 검사
7) 작업 순서 탐색(local search): EDD 순서에서 시작해 swap/insert 이동으로
   J = 총지연 + 지연건수 패널티 + 에너지비용 가중합 을 개선 (재현 가능하도록 seed 고정)
8) numpy 슬롯 배열 기반으로 구현해 기존 대비 수십 배 빠름 → 탐색/튜닝 가능

출력: slack_aware_energy_v2_schedule.csv, slack_aware_energy_v2_metrics.json

실행 예:
python src/slack_aware_energy_scheduler_v2.py --input-dir ./virtual_data --output-dir ./output_uci --iterations 1500
"""
import argparse
import json
import math
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

SLOT_MIN = 15
SETUP_POWER_RATIO = 0.2

DEFAULT_PARAMS = {
    # --- 시간 이동(에너지 최적화) 정책 ---
    "shift_load_types": ["Maximum"],   # 시간 이동 대상 Load_Type
    "min_slack_h": 8.0,                # 이 slack 미만이면 최조 배정 (EDD식)
    "slack_use_ratio": 0.5,            # slack 중 이동에 쓸 수 있는 비율
    "max_window_h": 24.0,              # 최대 이동 범위
    "bottleneck_threshold": 1.0,       # 부하율이 이 값 이상이면 병목 설비
    "bottleneck_window_h": 0.0,        # 병목 설비 최대 이동 범위
    # --- 후보 점수 가중치 (원 단위로 환산) ---
    "w_peak_slot": 3000.0,             # 피크 슬롯 1개(15분)당 패널티
    "w_demand_kW_slot": 50.0,          # 수요상한 초과 kW x 슬롯당 패널티
    "w_delay_min": 30.0,               # 최조 시작 대비 지연 1분당 패널티
    "demand_cap_kW": 650.0,            # 공장 전체(UCI 기저 + 라인) 수요전력 상한
    # --- 작업 순서 탐색 ---
    "iterations": 1500,              # 약 2분. 시드 42/7/123 모두 300회 대비 총지연 약 7% 추가 감소
    "seed": 42,
    "obj_tardy_job_min": 600.0,        # 지연 작업 1건 = 지연 600분과 동일 취급
    "obj_cost_per_min": 0.02,          # 에너지비용 50원 = 지연 1분 (납기가 에너지보다 우선)
}


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
class Problem:
    def __init__(self, jobs, machines, routing, maint, context):
        ctx = context.copy()
        ctx["time_slot"] = pd.to_datetime(ctx["time_slot"])
        ctx = ctx.sort_values("time_slot").reset_index(drop=True)
        self.t0 = ctx["time_slot"].iloc[0]
        self.H = len(ctx)
        self.price = ctx["price_per_kWh"].to_numpy(float)
        self.peak = ctx["peak_flag"].to_numpy(float)
        self.base = ctx["base_load_kW"].to_numpy(float) if "base_load_kW" in ctx else np.zeros(self.H)
        self.c_price = np.concatenate([[0], np.cumsum(self.price)])
        self.c_peak = np.concatenate([[0], np.cumsum(self.peak)])
        tod = (ctx["time_slot"].dt.hour * 60 + ctx["time_slot"].dt.minute).to_numpy()

        self.machines = machines.set_index("machine_id")
        self.mids = list(self.machines.index)
        self.avail = {}
        for mid, row in self.machines.iterrows():
            s = int(row.available_start_time[:2]) * 60 + int(row.available_start_time[3:])
            e = 24 * 60 if row.available_end_time == "23:59" else int(row.available_end_time[:2]) * 60 + int(row.available_end_time[3:])
            self.avail[mid] = (tod >= s) & (tod < e)
        maint = maint.copy()
        for c in ["maintenance_start", "maintenance_end"]:
            maint[c] = pd.to_datetime(maint[c])
        for _, r in maint.iterrows():
            a, b = self.floor_idx(r.maintenance_start), self.ceil_idx(r.maintenance_end)
            self.avail[r.machine_id][max(a, 0):min(b, self.H)] = False

        jobs = jobs.copy()
        for c in ["release_time", "due_time"]:
            jobs[c] = pd.to_datetime(jobs[c])
        self.jobs = jobs.set_index("job_id")
        self.job_ids = list(jobs["job_id"])
        self.products = sorted(routing.product_type.unique())
        self.pcode = {p: i for i, p in enumerate(self.products)}

        # 작업별 공정 정보 사전 계산
        self.ops = {}
        for jid, job in self.jobs.iterrows():
            steps = []
            for _, rt in routing[routing.product_type == job.product_type].sort_values("process_step").iterrows():
                m = self.machines.loc[rt.eligible_machine_id]
                cap_min = job.quantity / m.capacity_per_hour * 60
                proc_min = int(math.ceil(max(rt.process_time_min, cap_min) / SLOT_MIN) * SLOT_MIN)
                steps.append({
                    "step": int(rt.process_step), "process_id": rt.process_id, "machine": rt.eligible_machine_id,
                    "load_type": rt.load_type, "proc_min": proc_min, "setup_min": int(rt.setup_time_min),
                    "power": float(m.operation_power_kW),
                })
            for i, s in enumerate(steps):
                s["downstream_slots"] = sum(x["proc_min"] for x in steps[i + 1:]) // SLOT_MIN
            self.ops[jid] = steps
        self.release_idx = {j: self.ceil_idx(self.jobs.loc[j, "release_time"]) for j in self.job_ids}
        self.due_min = {j: (self.jobs.loc[j, "due_time"] - self.t0).total_seconds() / 60 for j in self.job_ids}

        # 설비 부하율 (계획 1주 기준 필요시간 / 가용시간)
        week = 7 * 24 * 60 // SLOT_MIN
        need = {m: 0.0 for m in self.mids}
        for steps in self.ops.values():
            for s in steps:
                need[s["machine"]] += s["proc_min"] / 60
        self.load_ratio = {m: need[m] / (self.avail[m][:week].sum() * SLOT_MIN / 60) for m in self.mids}

    def floor_idx(self, ts):
        return int(math.floor((pd.Timestamp(ts) - self.t0) / pd.Timedelta(minutes=SLOT_MIN)))

    def ceil_idx(self, ts):
        return int(math.ceil((pd.Timestamp(ts) - self.t0) / pd.Timedelta(minutes=SLOT_MIN)))

    def ts(self, idx):
        return self.t0 + pd.Timedelta(minutes=SLOT_MIN * idx)


# ---------------------------------------------------------------------------
# 스케줄 생성 (주어진 작업 순서 → 시뮬레이션)
# ---------------------------------------------------------------------------
def feasible_starts(free, ready, d):
    """free[t:t+d]가 모두 True인 t >= ready 목록 (numpy)."""
    f = free[ready:].astype(np.int32)
    if len(f) < d:
        return np.empty(0, int)
    cs = np.concatenate([[0], np.cumsum(f)])
    ok = (cs[d:] - cs[:-d]) == d
    return np.nonzero(ok)[0] + ready


def simulate(P, order, params, record=False):
    shift_types = set(params["shift_load_types"])
    busy = {m: np.zeros(P.H, bool) for m in P.mids}
    timeline = {m: [] for m in P.mids}       # 설비별 [(start, end_slot, product_code, setup_min)] 정렬 유지
    line = np.zeros(P.H)
    total_cost, total_tard, tardy, total_peak = 0.0, 0.0, 0, 0
    rows = []
    slot_h = SLOT_MIN / 60

    for jid in order:
        prod = P.pcode[P.jobs.loc[jid, "product_type"]]
        ready = P.release_idx[jid]
        due_min = P.due_min[jid]
        end_min = None
        for op in P.ops[jid]:
            m = op["machine"]
            free = P.avail[m] & ~busy[m]
            tl = timeline[m]
            starts_tl = [x[0] for x in tl]
            d0 = op["proc_min"] // SLOT_MIN
            d1 = int(math.ceil((op["proc_min"] + op["setup_min"]) / SLOT_MIN))

            def valid(t, d, with_setup):
                """시간상 직전/직후 공정과의 셋업 정합성 검사."""
                k = bisect_right(starts_tl, t)
                pred_prod = tl[k - 1][2] if k > 0 else None
                need_setup = pred_prod is not None and pred_prod != prod
                if need_setup != with_setup:
                    return False
                if k < len(tl):  # 직후 공정이 있으면: 같은 제품이거나 이미 셋업을 포함해야 함
                    nxt = tl[k]
                    if nxt[2] != prod and nxt[3] == 0:
                        return False
                return True

            # 후보: (start, slots, setup_min)
            # 최조 유효 시작 이후 max_window_h 범위까지만 수집 (속도)
            cands = []
            limit = None
            # 이동 대상이 아닌 공정은 최조 유효 후보만 필요
            max_win_slots = int(params["max_window_h"] * 60 // SLOT_MIN) if op["load_type"] in shift_types else 0
            for d, with_setup in ((d0, False), (d1, True)):
                for t in feasible_starts(free, ready, d):
                    t = int(t)
                    if limit is not None and t > limit:
                        break
                    if valid(t, d, with_setup):
                        cands.append((t, d, op["setup_min"] if with_setup else 0))
                        if limit is None or t + max_win_slots < limit:
                            limit = t + max_win_slots
            if not cands:
                raise RuntimeError(f"No feasible slot for {jid} step {op['step']} (horizon too short?)")
            cands.sort()
            t_e, d_e, _ = cands[0]

            # 하류공정 인지 slack (분)
            slack_min = due_min - (t_e + d_e + op["downstream_slots"]) * SLOT_MIN
            window = 0
            mode = "earliest"
            if op["load_type"] in shift_types and slack_min >= params["min_slack_h"] * 60:
                win_h = min(params["max_window_h"], slack_min / 60 * params["slack_use_ratio"])
                if P.load_ratio[m] >= params["bottleneck_threshold"]:
                    win_h = min(win_h, params["bottleneck_window_h"])
                    mode = "bottleneck_limited"
                else:
                    mode = "energy_shift"
                window = int(win_h * 60 // SLOT_MIN)

            if window > 0:
                p = op["power"]
                cap = params["demand_cap_kW"]
                excess = np.maximum(0, P.base + line + p - cap)
                c_ex = np.concatenate([[0], np.cumsum(excess)])
                best, best_score = None, None
                for t, d, su in cands:
                    if t > t_e + window:
                        break
                    s_slots = int(math.ceil(su / SLOT_MIN))
                    cost = (p * SETUP_POWER_RATIO * slot_h * (P.c_price[t + s_slots] - P.c_price[t])
                            + p * slot_h * (P.c_price[t + d] - P.c_price[t + s_slots]))
                    peak = P.c_peak[t + d] - P.c_peak[t]
                    score = (cost + params["w_peak_slot"] * peak
                             + params["w_demand_kW_slot"] * (c_ex[t + d] - c_ex[t])
                             + params["w_delay_min"] * (t - t_e) * SLOT_MIN
                             + 8.0 * su)  # 셋업 최소화 (v1과 동일 가중)
                    if best_score is None or score < best_score - 1e-9:
                        best, best_score = (t, d, su), score
                t, d, su = best
            else:
                t, d, su = cands[0]

            # 배정
            busy[m][t:t + d] = True
            k = bisect_right(starts_tl, t)
            tl.insert(k, (t, t + d, prod, su))
            s_slots = int(math.ceil(su / SLOT_MIN))
            p = op["power"]
            line[t:t + s_slots] += p * SETUP_POWER_RATIO
            line[t + s_slots:t + d] += p
            cost = (p * SETUP_POWER_RATIO * slot_h * (P.c_price[t + s_slots] - P.c_price[t])
                    + p * slot_h * (P.c_price[t + d] - P.c_price[t + s_slots]))
            total_cost += cost
            total_peak += int(P.c_peak[t + d] - P.c_peak[t])
            end_min = t * SLOT_MIN + su + op["proc_min"]
            ready = t + d
            if record:
                rows.append({
                    "job_id": jid, "product_type": P.products[prod],
                    "quantity": int(P.jobs.loc[jid, "quantity"]), "priority": int(P.jobs.loc[jid, "priority"]),
                    "due_time": P.jobs.loc[jid, "due_time"], "process_step": op["step"],
                    "process_id": op["process_id"], "machine_id": m, "load_type": op["load_type"],
                    "start_time": P.ts(t), "end_time": P.t0 + pd.Timedelta(minutes=end_min),
                    "setup_time_min": su, "processing_time_min": op["proc_min"],
                    "total_time_min": su + op["proc_min"], "energy_cost_est": round(cost, 2),
                    "peak_overlap_slots": int(P.c_peak[t + d] - P.c_peak[t]),
                    "slack_before_operation_min": round(slack_min, 1), "slack_policy_mode": mode,
                    "shift_min": (t - t_e) * SLOT_MIN,
                })
        tard = max(0.0, end_min - due_min)
        total_tard += tard
        tardy += tard > 0

    obj = total_tard + params["obj_tardy_job_min"] * tardy + params["obj_cost_per_min"] * total_cost
    return {"objective": obj, "total_tardiness_min": total_tard, "tardy_jobs": tardy,
            "energy_cost": total_cost, "peak_slots": total_peak, "rows": rows}


# ---------------------------------------------------------------------------
# 작업 순서 local search
# ---------------------------------------------------------------------------
def edd_order(P):
    j = P.jobs.reset_index()
    return list(j.sort_values(["due_time", "priority", "release_time", "job_id"])["job_id"])


def local_search(P, params, log=print):
    rng = np.random.default_rng(params["seed"])
    order = edd_order(P)
    cur = simulate(P, order, params)
    best_order, best = order[:], cur
    n = len(order)
    t0 = time.time()
    for it in range(1, params["iterations"] + 1):
        cand = best_order[:]
        if rng.random() < 0.5:   # 인접/근접 swap
            i = int(rng.integers(0, n - 1))
            j = min(n - 1, i + int(rng.integers(1, 6)))
            cand[i], cand[j] = cand[j], cand[i]
        else:                    # 제거 후 근처에 삽입
            i = int(rng.integers(0, n))
            job = cand.pop(i)
            j = int(np.clip(i + rng.integers(-10, 11), 0, n - 1))
            cand.insert(j, job)
        res = simulate(P, cand, params)
        if res["objective"] < best["objective"] - 1e-9:
            best_order, best = cand, res
        if it % 100 == 0:
            log(f"  iter {it:4d}  obj={best['objective']:.0f}  tardiness={best['total_tardiness_min']:.0f}  "
                f"tardy={best['tardy_jobs']}  cost={best['energy_cost']:.0f}  ({time.time() - t0:.0f}s)")
    return best_order, best


def build_schedule(jobs, machines, routing, maint, context, params=None, log=print):
    params = {**DEFAULT_PARAMS, **(params or {})}
    P = Problem(jobs, machines, routing, maint, context)
    log("machine load ratio: " + ", ".join(f"{m}={r:.2f}" for m, r in P.load_ratio.items()))
    base = simulate(P, edd_order(P), params)
    log(f"EDD order + v2 placement: tardiness={base['total_tardiness_min']:.0f} tardy={base['tardy_jobs']} cost={base['energy_cost']:.0f}")
    order, _ = local_search(P, params, log) if params["iterations"] > 0 else (edd_order(P), base)
    final = simulate(P, order, params, record=True)
    sched = pd.DataFrame(final["rows"])
    sched["sequence_rank"] = sched["job_id"].map({j: i for i, j in enumerate(order)})
    return sched, final, params, P


def compute_metrics(schedule, job_orders):
    job_orders = job_orders.copy()
    job_orders["due_time"] = pd.to_datetime(job_orders["due_time"])
    due = job_orders.set_index("job_id")["due_time"].to_dict()
    completion = pd.to_datetime(schedule.groupby("job_id")["end_time"].max())
    tard = completion.index.to_series().map(lambda j: max(0, (completion[j] - due[j]).total_seconds() / 60))
    return {
        "scheduled_operations": int(len(schedule)),
        "scheduled_jobs": int(completion.shape[0]),
        "on_time_jobs": int((tard == 0).sum()),
        "tardy_jobs": int((tard > 0).sum()),
        "due_date_adherence_rate": round(float((tard == 0).mean()), 4),
        "total_tardiness_min": round(float(tard.sum()), 2),
        "makespan_end": str(pd.to_datetime(schedule.end_time).max()),
        "total_setup_time_min": int(schedule.setup_time_min.sum()),
        "total_energy_cost_est": round(float(schedule.energy_cost_est.sum()), 2),
        "total_peak_overlap_slots": int(schedule.peak_overlap_slots.sum()),
        "policy_mode_counts": schedule["slack_policy_mode"].value_counts().to_dict(),
        "shifted_operations": int((schedule.shift_min > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="./virtual_data")
    ap.add_argument("--output-dir", default="./output_uci")
    ap.add_argument("--iterations", type=int, default=None, help="작업 순서 탐색 반복 수 (0이면 EDD 순서 고정)")
    ap.add_argument("--params", default=None, help="파라미터 JSON 파일 경로 (DEFAULT_PARAMS 덮어쓰기)")
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ctx_path = inp / "energy_context.csv"
    if not ctx_path.exists():
        raise SystemExit("energy_context.csv가 없습니다. 먼저 src/uci_energy_integration.py를 실행하세요.")
    params = {}
    if args.params:
        params.update(json.load(open(args.params, encoding="utf-8")))
    if args.iterations is not None:
        params["iterations"] = args.iterations

    jobs = pd.read_csv(inp / "job_orders.csv")
    sched, final, used, P = build_schedule(
        jobs, pd.read_csv(inp / "machines.csv"), pd.read_csv(inp / "product_routing.csv"),
        pd.read_csv(inp / "maintenance_schedule.csv"), pd.read_csv(ctx_path), params)
    sched.to_csv(out / "slack_aware_energy_v2_schedule.csv", index=False, encoding="utf-8-sig")
    metrics = compute_metrics(sched, jobs)
    metrics["params"] = used
    metrics["machine_load_ratio"] = {m: round(r, 3) for m, r in P.load_ratio.items()}
    with open(out / "slack_aware_energy_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "params"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
