"""
Common metrics module for production schedules.
모든 스케줄(FIFO, EDD, Energy-aware v1, Slack-aware, Slack-aware v2)을 같은 규칙으로 재평가한다.

v2 변경점 (UCI 결합)
- energy_context.csv(6주, 15분)를 사용 → 기존 1주 요금표 밖(9/28 이후) 공정도 실제 TOU 단가·피크로 평가
- CO2 = kWh x UCI에서 추정한 배출계수(kgCO2/kWh). 기존 Load_Type 프록시 값도 비교용으로 함께 출력
- 셋업 구간은 가동전력의 20%로 계산 (스케줄러 내부 가정과 통일)
- UCI 기저부하 + 라인 부하 = 공장 전체 수요전력(facility demand) 지표 추가
- 설비 대기전력(idle) 에너지 지표 추가: 설비의 첫 작업~마지막 작업 사이, 가동 가능 시간 중 비가동 슬롯 x idle_power
"""
import json

import numpy as np
import pandas as pd

SLOT_MIN = 15
SETUP_POWER_RATIO = 0.2
LOAD_CO2_FACTOR = {"Light": 0.35, "Medium": 0.45, "Maximum": 0.60}  # 기존 프록시 (비교용)
DEFAULT_CO2_FACTOR = 0.4585  # UCI 추정치 (energy_context가 없을 때만 사용)


def _to_dt(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def context_from_tariff(tariff):
    """energy_context가 없을 때 기존 tou_tariff로 최소 context 생성 (기저부하 0)."""
    ctx = _to_dt(tariff, ["time_slot"])[["time_slot", "tariff_type", "price_per_kWh", "peak_flag"]].copy()
    ctx["base_load_kW"] = 0.0
    ctx["co2_factor_kg_per_kWh"] = DEFAULT_CO2_FACTOR
    return ctx


class SlotGrid:
    """15분 슬롯 배열 기반 빠른 조회용 헬퍼."""

    def __init__(self, context):
        ctx = _to_dt(context, ["time_slot"]).sort_values("time_slot").reset_index(drop=True)
        self.ctx = ctx
        self.t0 = ctx["time_slot"].iloc[0]
        self.n = len(ctx)
        self.price = ctx["price_per_kWh"].to_numpy(float)
        self.peak = ctx["peak_flag"].to_numpy(int)
        self.base = ctx["base_load_kW"].to_numpy(float) if "base_load_kW" in ctx else np.zeros(self.n)
        self.co2f = (ctx["co2_factor_kg_per_kWh"].to_numpy(float)
                     if "co2_factor_kg_per_kWh" in ctx else np.full(self.n, DEFAULT_CO2_FACTOR))

    def idx(self, ts):
        return int((pd.Timestamp(ts) - self.t0) / pd.Timedelta(minutes=SLOT_MIN))

    def span(self, start, end):
        """[start, end) 가 걸치는 슬롯 인덱스 범위 (start는 슬롯 내림)."""
        a = self.idx(pd.Timestamp(start).floor(f"{SLOT_MIN}min"))
        b = self.idx(pd.Timestamp(end).ceil(f"{SLOT_MIN}min"))
        return a, b


def _slot_weights(grid, start, end):
    """[start, end) 구간이 각 슬롯을 차지하는 비율(0~1). 반환: (a, b, weights[b-a])."""
    a, b = grid.span(start, end)
    if b <= a:
        return a, a, np.zeros(0)
    # 분 단위 정수로 계산 (pandas 버전과 무관하게 동작)
    s_min = (pd.Timestamp(start) - grid.t0).total_seconds() / 60
    e_min = (pd.Timestamp(end) - grid.t0).total_seconds() / 60
    slot_s = np.arange(a, b) * SLOT_MIN
    w = (np.minimum(slot_s + SLOT_MIN, e_min) - np.maximum(slot_s, s_min)) / SLOT_MIN
    w = np.clip(w, 0, 1)
    return a, b, np.asarray(w, float)


def _op_power_series(schedule, machines, grid):
    """각 공정을 (row, machine, slot_from, slot_to, power_kW, weights) 세그먼트로 분해.

    셋업 구간은 가동전력의 20%, 가공 구간은 100%.
    셋업 종료가 슬롯 중간(예: 20분)에 걸리면 슬롯 점유 비율(weights)로 나눠 계산한다
    (이전 버전은 경계 슬롯을 셋업·가공 양쪽에 중복 계산해 수요전력이 과대평가됨).
    """
    m_power = machines.set_index("machine_id")["operation_power_kW"].to_dict()
    segs = []
    for i, r in schedule.iterrows():
        p = float(m_power.get(r["machine_id"], 0))
        setup = int(r.get("setup_time_min", 0) or 0)
        start, end = r["start_time"], r["end_time"]
        setup_end = min(start + pd.Timedelta(minutes=setup), end)
        if setup > 0:
            a, b, w = _slot_weights(grid, start, setup_end)
            segs.append((i, r["machine_id"], a, b, p * SETUP_POWER_RATIO, w))
        a, b, w = _slot_weights(grid, setup_end, end)
        segs.append((i, r["machine_id"], a, b, p, w))
    return segs


def _clip(a, b, w, n):
    """context 범위 밖 슬롯을 잘라낸다."""
    lo, hi = max(a, 0), min(b, n)
    if hi <= lo:
        return lo, lo, w[:0]
    return lo, hi, w[lo - a:hi - a]


def normalize_schedule(schedule, machines, tariff=None, context=None):
    """스케줄별 공정 단위 공통 에너지/비용/피크/CO2 컬럼을 계산."""
    schedule = _to_dt(schedule, ["start_time", "end_time", "due_time"]).reset_index(drop=True)
    grid = SlotGrid(context if context is not None else context_from_tariff(tariff))
    n_ops = len(schedule)
    kwh = np.zeros(n_ops); cost = np.zeros(n_ops); peak = np.zeros(n_ops, int)
    co2 = np.zeros(n_ops); out_of_ctx = np.zeros(n_ops, int)
    slot_h = SLOT_MIN / 60
    for i, _mid, a, b, p, w in _op_power_series(schedule, machines, grid):
        if a < 0 or b > grid.n:
            out_of_ctx[i] = 1
        a, b, w = _clip(a, b, w, grid.n)
        e = p * slot_h
        kwh[i] += e * w.sum()
        cost[i] += e * (w * grid.price[a:b]).sum()
        co2[i] += e * (w * grid.co2f[a:b]).sum()
    # 피크 겹침 슬롯: 공정 전체 구간 [start, end)가 걸치는 피크 슬롯 수 (셋업/가공 경계 중복 없음)
    for i, r in schedule.iterrows():
        a, b = grid.span(r["start_time"], r["end_time"])
        peak[i] = int(grid.peak[max(a, 0):min(b, grid.n)].sum())
    schedule["common_energy_kWh"] = kwh.round(3)
    schedule["common_energy_cost"] = cost.round(2)
    schedule["common_peak_overlap_slots"] = peak
    schedule["common_co2_kg"] = co2.round(3)
    schedule["common_co2_kg_loadtype_proxy"] = (kwh * schedule["load_type"].map(LOAD_CO2_FACTOR).fillna(0.45)).round(3)
    schedule["outside_energy_context"] = out_of_ctx
    return schedule


def build_power_profile(schedule, machines, tariff=None, context=None):
    """15분 슬롯 단위 라인 전력 + UCI 기저부하 = 공장 전체 전력 프로파일."""
    schedule = _to_dt(schedule, ["start_time", "end_time"]).reset_index(drop=True)
    grid = SlotGrid(context if context is not None else context_from_tariff(tariff))
    line = np.zeros(grid.n)
    for _i, _mid, a, b, p, w in _op_power_series(schedule, machines, grid):
        a, b, w = _clip(a, b, w, grid.n)
        line[a:b] += p * w  # 15분 평균 전력 (수요전력 산정 기준과 동일)
    profile = grid.ctx[["time_slot", "tariff_type", "price_per_kWh", "peak_flag"]].copy()
    profile["line_power_kW"] = line
    profile["base_load_kW"] = grid.base
    profile["facility_power_kW"] = line + grid.base
    profile["total_power_kW"] = line  # 하위호환 (라인 전력)
    profile["energy_kWh"] = line * (SLOT_MIN / 60)
    profile["energy_cost"] = profile["energy_kWh"] * profile["price_per_kWh"]
    return profile


def idle_energy(schedule, machines, maintenance, grid):
    """설비별 대기전력 에너지: 첫 작업 시작~마지막 작업 종료 사이 가동가능 & 비가동 슬롯."""
    m = machines.set_index("machine_id")
    maint = _to_dt(maintenance, ["maintenance_start", "maintenance_end"]) if maintenance is not None else None
    slot_times = grid.ctx["time_slot"]
    tod = (slot_times.dt.hour * 60 + slot_times.dt.minute).to_numpy()
    total, per_machine = 0.0, {}
    for mid, g in schedule.groupby("machine_id"):
        row = m.loc[mid]
        s_h, s_m = map(int, str(row["available_start_time"]).split(":"))
        e_str = str(row["available_end_time"])
        e_min = 24 * 60 if e_str == "23:59" else int(e_str[:2]) * 60 + int(e_str[3:])
        avail = (tod >= s_h * 60 + s_m) & (tod < e_min)
        if maint is not None:
            for _, r in maint[maint.machine_id == mid].iterrows():
                a, b = grid.span(r.maintenance_start, r.maintenance_end)
                avail[max(a, 0):min(b, grid.n)] = False
        busy = np.zeros(grid.n, bool)
        for _, r in g.iterrows():
            a, b = grid.span(r.start_time, r.end_time)
            busy[max(a, 0):min(b, grid.n)] = True
        a0, b0 = grid.span(g.start_time.min(), g.end_time.max())
        idle_slots = int((avail & ~busy)[max(a0, 0):min(b0, grid.n)].sum())
        kwh = idle_slots * float(row["idle_power_kW"]) * SLOT_MIN / 60
        per_machine[mid] = round(kwh, 2)
        total += kwh
    return round(total, 3), per_machine


def calculate_metrics(schedule, job_orders, machines, tariff=None, schedule_name="schedule",
                      context=None, maintenance=None):
    job_orders = _to_dt(job_orders, ["release_time", "due_time"])
    ctx = context if context is not None else context_from_tariff(tariff)
    grid = SlotGrid(ctx)
    schedule = normalize_schedule(schedule, machines, context=ctx)
    completion = schedule.groupby("job_id")["end_time"].max()
    due = job_orders.set_index("job_id")["due_time"].to_dict()
    tardiness = completion.index.to_series().map(lambda j: max(0, (completion[j] - due[j]).total_seconds() / 60))
    profile = build_power_profile(schedule, machines, context=ctx)
    idle_kwh, idle_by_machine = idle_energy(schedule, machines, maintenance, grid)

    machine_util = []
    for mid, g in schedule.groupby("machine_id"):
        busy_h = ((g["end_time"] - g["start_time"]).dt.total_seconds() / 3600).sum()
        machine_util.append({"machine_id": mid, "busy_hours": round(float(busy_h), 2),
                             "idle_energy_kWh": idle_by_machine.get(mid, 0.0)})

    active = profile[(profile.time_slot >= schedule.start_time.min()) & (profile.time_slot < schedule.end_time.max())]
    peak_rows = profile[profile.peak_flag == 1]
    co2_factor = float(np.mean(grid.co2f))

    op_kwh = float(schedule["common_energy_kWh"].sum())
    metrics = {
        "schedule_name": schedule_name,
        "scheduled_operations": int(len(schedule)),
        "scheduled_jobs": int(completion.shape[0]),
        "on_time_jobs": int((tardiness == 0).sum()),
        "tardy_jobs": int((tardiness > 0).sum()),
        "due_date_adherence_rate": round(float((tardiness == 0).mean()), 4),
        "total_tardiness_min": round(float(tardiness.sum()), 2),
        "avg_tardiness_min": round(float(tardiness.mean()), 2),
        "max_tardiness_min": round(float(tardiness.max()), 2),
        "makespan_start": str(schedule["start_time"].min()),
        "makespan_end": str(schedule["end_time"].max()),
        "total_setup_time_min": int(schedule["setup_time_min"].sum()) if "setup_time_min" in schedule else 0,
        "total_energy_kWh": round(op_kwh, 3),
        "idle_energy_kWh": idle_kwh,
        "total_energy_incl_idle_kWh": round(op_kwh + idle_kwh, 3),
        "total_energy_cost": round(float(schedule["common_energy_cost"].sum()), 2),
        "avg_price_per_kWh": round(float(schedule["common_energy_cost"].sum()) / op_kwh, 2) if op_kwh else 0,
        "total_peak_overlap_slots": int(schedule["common_peak_overlap_slots"].sum()),
        "peak_period_energy_kWh": round(float(peak_rows["energy_kWh"].sum()), 3),
        "peak_period_energy_share": round(float(peak_rows["energy_kWh"].sum()) / op_kwh, 4) if op_kwh else 0,
        "max_demand_kW": round(float(profile["line_power_kW"].max()), 3),
        "facility_max_demand_kW": round(float(active["facility_power_kW"].max()), 3),
        "facility_peak_time_max_demand_kW": round(float(peak_rows["facility_power_kW"].max()), 3),
        "total_co2_kg": round(float(schedule["common_co2_kg"].sum()), 3),
        "total_co2_incl_idle_kg": round(float(schedule["common_co2_kg"].sum()) + idle_kwh * co2_factor, 3),
        "total_co2_kg_loadtype_proxy": round(float(schedule["common_co2_kg_loadtype_proxy"].sum()), 3),
        "ops_outside_energy_context": int(schedule["outside_energy_context"].sum()),
        "machine_busy_hours": machine_util,
    }
    return metrics, schedule, profile


def save_metrics(metrics, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
