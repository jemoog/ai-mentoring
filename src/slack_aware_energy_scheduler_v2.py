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
9) (5차시) 대체 설비: 공정 단계마다 여러 설비 옵션을 비교. 기본은 가장 빨리 끝나는 설비,
   energy_shift 대상이면 전력비까지 고려해 선택 (비효율 대체 설비는 납기 여유가 없을 때 주로 사용)
10) (5차시) 최대수요전력 제어 (--demand-mode, 기본 꺼짐): 배정 시 공장 최대수요(UCI 기저부하 예측 + 라인)
   증가분에 패널티, slack 8시간 이상·비병목 설비 공정만 최대 2시간 이동.
   기본요금(최대수요)은 줄지만 전력량요금이 늘 수 있어 선택 옵션으로 둠

출력: slack_aware_energy_v2_schedule.csv, slack_aware_energy_v2_metrics.json

실행 예:
python src/slack_aware_energy_scheduler_v2.py --input-dir ./virtual_data --output-dir ./output_uci --iterations 1500
python src/slack_aware_energy_scheduler_v2.py --input-dir ./virtual_data --output-dir ./output_uci --demand-mode
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
    "min_slack_h": 24.0,               # 이 slack 미만이면 최조 배정 (EDD식)  ※시나리오 실험으로 8→24 조정
    "slack_use_ratio": 0.5,            # slack 중 이동에 쓸 수 있는 비율
    "max_window_h": 24.0,              # 최대 이동 범위
    "bottleneck_threshold": 0.8,       # 부하율이 이 값 이상이면 병목 설비  ※시나리오 실험으로 1.0→0.8 조정
                                       #   (부하율 85~100% 설비에서 시간 이동 시 다른 작업 연쇄 지연 확인)
    "bottleneck_window_h": 0.0,        # 병목 설비 최대 이동 범위
    # --- 후보 점수 가중치 (원 단위로 환산) ---
    "w_peak_slot": 3000.0,             # 피크 슬롯 1개(15분)당 패널티
    "w_delay_min": 30.0,               # 최조 완료 대비 지연 1분당 패널티
    # --- 최대수요전력 제어 (공장 전체 = UCI 기저부하 예측 + 라인) ---
    "w_demand_kW": 0.0,                # 배정 시 공장 최대수요 증가 1kW당 패널티(원). 0이면 끔
    "demand_window_h": 2.0,            # 최대수요 회피용 최대 이동 범위 (모든 공정)
    "demand_min_slack_h": 8.0,         # 이 slack 이상일 때만 최대수요 회피 이동 허용
    "obj_demand_min_per_kW": 0.0,      # 순서 탐색 목적함수: 최대수요 1kW 증가 = 지연 n분
    # ※ 최대수요 제어는 기본 꺼짐. 켜면 기본요금(최대수요)은 줄지만 일부 작업이 저녁 피크요금 시간대로
    #   이동해 전력량요금이 늘 수 있다 → DEMAND_MODE 프리셋 / --demand-mode 옵션으로 선택
    # --- 작업 순서 탐색 ---
    "iterations": 1500,              # 약 2분. 시드 42/7/123 모두 300회 대비 총지연 약 7% 추가 감소
    "seed": 42,
    "obj_tardy_job_min": 600.0,        # 지연 작업 1건 = 지연 600분과 동일 취급
    "obj_cost_per_min": 0.02,          # 에너지비용 50원 = 지연 1분 (납기가 에너지보다 우선)
}


# 최대수요 제어 모드 프리셋 (시나리오 실험으로 선택: 부하율 70%에서 공장 최대수요 약 -6%)
DEMAND_MODE = {"w_demand_kW": 8000.0, "demand_window_h": 2.0, "demand_min_slack_h": 8.0,
               "obj_demand_min_per_kW": 0.0}


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

        # 작업별 공정 정보 사전 계산 (공정 단계마다 대체 설비 옵션 목록)
        self.ops = {}
        for jid, job in self.jobs.iterrows():
            steps = []
            rts = routing[routing.product_type == job.product_type].sort_values("process_step")
            for step_no, grp in rts.groupby("process_step", sort=True):
                options = []
                for rt in grp.itertuples():
                    m = self.machines.loc[rt.eligible_machine_id]
                    cap_min = job.quantity / m.capacity_per_hour * 60
                    proc_min = int(math.ceil(max(rt.process_time_min, cap_min) / SLOT_MIN) * SLOT_MIN)
                    options.append({"machine": rt.eligible_machine_id, "process_id": rt.process_id,
                                    "load_type": rt.load_type, "proc_min": proc_min,
                                    "setup_min": int(rt.setup_time_min), "power": float(m.operation_power_kW)})
                steps.append({"step": int(step_no), "options": options,
                              "load_type": options[0]["load_type"],
                              "min_proc": min(o["proc_min"] for o in options)})
            for i, st in enumerate(steps):
                st["downstream_slots"] = sum(x["min_proc"] for x in steps[i + 1:]) // SLOT_MIN
            self.ops[jid] = steps
        # (제품코드, 설비) → 셋업시간(분), 설비 가동전력
        self.setup_min = {(self.pcode[r.product_type], r.eligible_machine_id): int(r.setup_time_min)
                          for r in routing.itertuples()}
        self.mpower = self.machines["operation_power_kW"].astype(float).to_dict()
        self.release_idx = {j: self.ceil_idx(self.jobs.loc[j, "release_time"]) for j in self.job_ids}
        self.due_min = {j: (self.jobs.loc[j, "due_time"] - self.t0).total_seconds() / 60 for j in self.job_ids}

        # 설비 부하율 (계획 1주 기준 필요시간 / 가용시간)
        # 대체 설비가 있으면 처리 속도(1/가공시간) 비율로 필요시간을 나눠 배분 (pooled 부하율)
        week = 7 * 24 * 60 // SLOT_MIN
        need = {m: 0.0 for m in self.mids}
        for steps in self.ops.values():
            for st in steps:
                speed = {o["machine"]: 1.0 / o["proc_min"] for o in st["options"]}
                tot = sum(speed.values())
                for o in st["options"]:
                    need[o["machine"]] += o["proc_min"] / 60 * speed[o["machine"]] / tot
        self.load_ratio = {m: need[m] / (self.avail[m][:week].sum() * SLOT_MIN / 60) for m in self.mids}
        # 공장 수요전력 기준선: 계획 1주 중 기저부하(UCI 평균 프로파일) 최대값
        self.base_week_max = float(self.base[:week].max())

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
    """주어진 작업 순서로 공정을 하나씩 배정하는 시뮬레이션.

    공정 단계마다
      1) 대체 설비 옵션별로 배정 가능한 시작 후보를 모은다 (셋업 정합성 검사 포함)
      2) 가장 빨리 끝나는 후보(완료 e_end)를 기준으로 하류공정 인지 slack을 계산
      3) 허용 범위(완료 지연 슬롯) 결정
         - energy_shift : 시간 이동 대상 Load_Type + slack 충분 + 비병목 설비 → 전력비·피크까지 고려
         - demand_shift : 최대수요 제어용 짧은 이동 (모든 공정, slack 충분할 때만) → 최대수요 증가분만 고려
         - earliest     : 가장 빨리 끝나는 후보 (동률이면 전력비 낮은 설비)
      4) 점수 최소 후보 배정
    """
    shift_types = set(params["shift_load_types"])
    busy = {m: np.zeros(P.H, bool) for m in P.mids}
    timeline = {m: [] for m in P.mids}       # 설비별 [(start, end_slot, product_code, setup_min)] 정렬 유지
    line = np.zeros(P.H)
    total_cost, total_tard, tardy, total_peak = 0.0, 0.0, 0, 0
    rows = []
    row_at = {}                               # (설비, 시작슬롯) → rows 인덱스 (셋업 추가 시 갱신용)
    slot_h = SLOT_MIN / 60
    run_max = P.base_week_max                 # 지금까지의 공장 최대수요 (기저부하 최대에서 시작)
    w_dem = params["w_demand_kW"]
    dem_win = int(params["demand_window_h"] * 60 // SLOT_MIN) if w_dem > 0 else 0
    energy_win = int(params["max_window_h"] * 60 // SLOT_MIN)

    def op_cost(t, d, su, p):
        ss = int(math.ceil(su / SLOT_MIN))
        return (p * SETUP_POWER_RATIO * slot_h * (P.c_price[t + ss] - P.c_price[t])
                + p * slot_h * (P.c_price[t + d] - P.c_price[t + ss]))

    def op_demand_peak(t, d, su, p):
        ss = int(math.ceil(su / SLOT_MIN))
        add = np.full(d, p)
        add[:ss] = p * SETUP_POWER_RATIO
        return float((P.base[t:t + d] + line[t:t + d] + add).max())

    for jid in order:
        prod = P.pcode[P.jobs.loc[jid, "product_type"]]
        ready = P.release_idx[jid]
        due_min = P.due_min[jid]
        end_min = None
        for st in P.ops[jid]:
            collect = max(energy_win if st["load_type"] in shift_types else 0, dem_win)

            # 1) 설비 옵션별 후보 수집: (start, slots, setup_min, fix, option)
            cands = []
            for opt in st["options"]:
                m = opt["machine"]
                free = P.avail[m] & ~busy[m]
                tl = timeline[m]
                starts_tl = [x[0] for x in tl]

                def check(t, d, with_setup, m=m, tl=tl, starts_tl=starts_tl):
                    """시간상 직전/직후 공정과의 셋업 정합성 검사.

                    반환: None(불가) / ()(가능) / (k, su_next, s_slots_next)(가능, 단 직후 공정에 셋업 추가 필요)
                    - 직전 공정 제품이 다르면 셋업 포함 후보만, 같으면(또는 없으면) 셋업 없는 후보만 허용
                    - 직후 공정이 다른 제품이고 셋업 없이 배정돼 있으면, 빈 구간 안에 그 공정의 셋업 시간을
                      확보할 수 있을 때만 허용 (셋업은 부품 도착 전에 하는 설비 준비 작업으로 간주)
                    """
                    k = bisect_right(starts_tl, t)
                    pred_prod = tl[k - 1][2] if k > 0 else None
                    need_setup = pred_prod is not None and pred_prod != prod
                    if need_setup != with_setup:
                        return None
                    if k < len(tl):
                        nxt = tl[k]
                        if nxt[2] != prod and nxt[3] == 0:
                            su_n = P.setup_min[(nxt[2], m)]
                            sn = int(math.ceil(su_n / SLOT_MIN))
                            if t + d + sn > nxt[0] or not P.avail[m][nxt[0] - sn:nxt[0]].all():
                                return None
                            return (k, su_n, sn)
                    return ()

                d0 = opt["proc_min"] // SLOT_MIN
                d1 = int(math.ceil((opt["proc_min"] + opt["setup_min"]) / SLOT_MIN))
                limit = None
                for d, with_setup in ((d0, False), (d1, True)):
                    for t in feasible_starts(free, ready, d):
                        t = int(t)
                        if limit is not None and t > limit:
                            break
                        fix = check(t, d, with_setup)
                        if fix is not None:
                            cands.append((t, d, opt["setup_min"] if with_setup else 0, fix, opt))
                            if limit is None or t + collect < limit:
                                limit = t + collect
            if not cands:
                raise RuntimeError(f"No feasible slot for {jid} step {st['step']} (horizon too short?)")

            # 2) 가장 빨리 끝나는 후보 기준 slack (분)
            e_end = min(c[0] + c[1] for c in cands)
            best_e = min((c for c in cands if c[0] + c[1] == e_end),
                         key=lambda c: (op_cost(c[0], c[1], c[2], c[4]["power"]), c[0]))
            slack_min = due_min - (e_end + st["downstream_slots"]) * SLOT_MIN

            # 3) 허용 범위 결정
            win, mode, energy_ok = 0, "earliest", False
            if st["load_type"] in shift_types and slack_min >= params["min_slack_h"] * 60:
                win_h = min(params["max_window_h"], slack_min / 60 * params["slack_use_ratio"])
                if P.load_ratio[best_e[4]["machine"]] >= params["bottleneck_threshold"]:
                    win_h = min(win_h, params["bottleneck_window_h"])
                    mode = "bottleneck_limited"
                else:
                    mode = "energy_shift"
                win = int(win_h * 60 // SLOT_MIN)
                energy_ok = win > 0
            # 최대수요 회피 이동도 비병목 설비에서만 (부하율 120~148% 실험에서 병목 이동 시 지연 +11%, 수요 감소 없음)
            if (dem_win > 0 and slack_min >= params["demand_min_slack_h"] * 60
                    and P.load_ratio[best_e[4]["machine"]] < params["bottleneck_threshold"]):
                dw = min(dem_win, int(slack_min * params["slack_use_ratio"] // SLOT_MIN))
                if dw > win:
                    win = dw
                    if not energy_ok:
                        mode = "demand_shift"

            # 4) 선택
            if win > 0:
                best, best_score = None, None
                for c in cands:
                    t, d, su, fix, opt = c
                    if t + d > e_end + win:
                        continue
                    p = opt["power"]
                    score = (params["w_delay_min"] * (t + d - e_end) * SLOT_MIN
                             + 8.0 * su + (8.0 * fix[1] if fix else 0.0))
                    if energy_ok:
                        score += op_cost(t, d, su, p) + params["w_peak_slot"] * (P.c_peak[t + d] - P.c_peak[t])
                    if w_dem > 0:
                        score += w_dem * max(0.0, op_demand_peak(t, d, su, p) - run_max)
                    if best_score is None or score < best_score - 1e-9:
                        best, best_score = c, score
                chosen = best
            else:
                chosen = best_e
            t, d, su, fix, opt = chosen
            m, p = opt["machine"], opt["power"]
            tl = timeline[m]
            starts_tl = [x[0] for x in tl]

            # 배정
            busy[m][t:t + d] = True
            k = bisect_right(starts_tl, t)
            tl.insert(k, (t, t + d, prod, su))
            if fix:  # 직후 공정(이제 k+1번째) 앞 빈 구간에 셋업 추가
                _, su_n, sn = fix
                n0, n1, n_prod, _ = tl[k + 1]
                tl[k + 1] = (n0 - sn, n1, n_prod, su_n)
                busy[m][n0 - sn:n0] = True
                pn = P.mpower[m] * SETUP_POWER_RATIO
                line[n0 - sn:n0] += pn
                add_cost = pn * slot_h * (P.c_price[n0] - P.c_price[n0 - sn])
                total_cost += add_cost
                run_max = max(run_max, float((P.base[n0 - sn:n0] + line[n0 - sn:n0]).max()))
                if record:
                    ri = row_at.pop((m, n0))
                    row_at[(m, n0 - sn)] = ri
                    rows[ri]["start_time"] = P.t0 + pd.Timedelta(minutes=n0 * SLOT_MIN - su_n)
                    rows[ri]["setup_time_min"] = su_n
                    rows[ri]["total_time_min"] += su_n
                    rows[ri]["energy_cost_est"] = round(rows[ri]["energy_cost_est"] + add_cost, 2)
                    rows[ri]["setup_added_later"] = True
            s_slots = int(math.ceil(su / SLOT_MIN))
            line[t:t + s_slots] += p * SETUP_POWER_RATIO
            line[t + s_slots:t + d] += p
            run_max = max(run_max, float((P.base[t:t + d] + line[t:t + d]).max()))
            cost = op_cost(t, d, su, p)
            total_cost += cost
            total_peak += int(P.c_peak[t + d] - P.c_peak[t])
            end_min = t * SLOT_MIN + su + opt["proc_min"]
            ready = t + d
            if record:
                row_at[(m, t)] = len(rows)
                rows.append({
                    "job_id": jid, "product_type": P.products[prod],
                    "quantity": int(P.jobs.loc[jid, "quantity"]), "priority": int(P.jobs.loc[jid, "priority"]),
                    "due_time": P.jobs.loc[jid, "due_time"], "process_step": st["step"],
                    "process_id": opt["process_id"], "machine_id": m, "load_type": opt["load_type"],
                    "start_time": P.ts(t), "end_time": P.t0 + pd.Timedelta(minutes=end_min),
                    "setup_time_min": su, "processing_time_min": opt["proc_min"],
                    "total_time_min": su + opt["proc_min"], "energy_cost_est": round(cost, 2),
                    "peak_overlap_slots": int(P.c_peak[t + d] - P.c_peak[t]),
                    "slack_before_operation_min": round(slack_min, 1), "slack_policy_mode": mode,
                    "shift_min": (t + d - e_end) * SLOT_MIN, "setup_added_later": False,
                    "n_machine_options": len(st["options"]),
                })
        tard = max(0.0, end_min - due_min)
        total_tard += tard
        tardy += tard > 0

    obj = (total_tard + params["obj_tardy_job_min"] * tardy + params["obj_cost_per_min"] * total_cost
           + params["obj_demand_min_per_kW"] * (run_max - P.base_week_max))
    return {"objective": obj, "total_tardiness_min": total_tard, "tardy_jobs": tardy,
            "energy_cost": total_cost, "peak_slots": total_peak, "facility_max_kW": run_max, "rows": rows}


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
        "operations_by_machine": schedule.machine_id.value_counts().sort_index().to_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="./virtual_data")
    ap.add_argument("--output-dir", default="./output_uci")
    ap.add_argument("--iterations", type=int, default=None, help="작업 순서 탐색 반복 수 (0이면 EDD 순서 고정)")
    ap.add_argument("--params", default=None, help="파라미터 JSON 파일 경로 (DEFAULT_PARAMS 덮어쓰기)")
    ap.add_argument("--demand-mode", action="store_true", help="최대수요전력 제어 켜기 (DEMAND_MODE 프리셋)")
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ctx_path = inp / "energy_context.csv"
    if not ctx_path.exists():
        raise SystemExit("energy_context.csv가 없습니다. 먼저 src/uci_energy_integration.py를 실행하세요.")
    params = {}
    if args.params:
        params.update(json.load(open(args.params, encoding="utf-8")))
    if args.demand_mode:
        params.update(DEMAND_MODE)
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
