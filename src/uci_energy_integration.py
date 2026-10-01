"""
UCI Steel Industry Energy Consumption Dataset 결합 모듈.

UCI 데이터(2018년, 국내 철강공장, 15분 단위 35,040행)에서 다음을 추출해
가상 생산 데이터의 계획 기간과 결합한다.

1) CO2 배출계수 추정
   - UCI의 CO2(tCO2) 컬럼은 Usage_kWh x 고정계수를 10kg 단위로 반올림한 값으로 확인됨
   - 반올림 일치율이 최대가 되는 계수를 탐색해 kgCO2/kWh 계수를 추정
   - (참고) Load_Type별 단순 비율은 저부하 구간의 반올림(0 처리) 때문에 낮게 왜곡됨
2) 요일 x 15분 슬롯 기준의 공장 기저부하(base load) 프로파일
   - 타 라인/공장 나머지 설비의 실제 전력 패턴으로 사용 → 공장 전체 수요전력 계산
3) 요일 x 15분 슬롯 기준의 UCI Load_Type (Light / Medium / Maximum)
   - 가상 TOU 요금 구분(경부하/중간부하/최대부하)과의 정합성 확인
4) 계획 기간 전체(기본 6주)를 덮는 energy_context.csv / tou_tariff_extended.csv 생성
   - 기존 tou_tariff.csv는 1주만 있어 9/28 이후 공정은 고정단가·피크 0으로 평가되던 문제 해결

실행 예:
python src/uci_energy_integration.py --uci-path ./uci_data/Steel_industry_data.csv \
    --data-dir ./virtual_data --uci-out-dir ./uci_data --weeks 6
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SLOT_MIN = 15
SLOTS_PER_DAY = 24 * 60 // SLOT_MIN
LOAD_TYPE_MAP = {"Light_Load": "Light", "Medium_Load": "Medium", "Maximum_Load": "Maximum"}
TARIFF_TO_LOAD = {"경부하": "Light", "중간부하": "Medium", "최대부하": "Maximum"}


# ---------------------------------------------------------------------------
# 1. UCI 로드 및 타임스탬프 정리
# ---------------------------------------------------------------------------
def load_uci(path):
    """UCI CSV 로드 + 타임스탬프 정리.

    - date는 'dd/mm/YYYY HH:MM' 형식이며 15분 구간의 '종료 시각'이다.
    - 자정(00:00) 행은 같은 날짜로 표기돼 있지만 실제로는 다음날 00:00(전날 23:45~24:00 구간)이다.
    - 따라서 00:00 행은 +1일 보정 후, 구간 시작 시각 = 종료 시각 - 15분 으로 변환한다.
    """
    df = pd.read_csv(path)
    ts_end = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    midnight = (ts_end.dt.hour == 0) & (ts_end.dt.minute == 0)
    ts_end = ts_end + pd.to_timedelta(midnight.astype(int), unit="D")
    df["slot_start"] = ts_end - pd.Timedelta(minutes=SLOT_MIN)
    df = df.sort_values("slot_start").reset_index(drop=True)
    df["load_type"] = df["Load_Type"].map(LOAD_TYPE_MAP)
    df["kW"] = df["Usage_kWh"] * (60 / SLOT_MIN)
    df["co2_kg"] = df["CO2(tCO2)"] * 1000
    df["weekday"] = df["slot_start"].dt.dayofweek  # 0=월
    df["slot_of_day"] = (df["slot_start"].dt.hour * 60 + df["slot_start"].dt.minute) // SLOT_MIN
    df["slot_of_week"] = df["weekday"] * SLOTS_PER_DAY + df["slot_of_day"]
    return df


# ---------------------------------------------------------------------------
# 2. CO2 배출계수 추정
# ---------------------------------------------------------------------------
def estimate_emission_factor(df):
    """CO2 = round(Usage_kWh * k, 10kg) 를 가장 잘 재현하는 k(kgCO2/kWh)를 탐색."""
    u = df["Usage_kWh"].to_numpy()
    c = df["co2_kg"].to_numpy()
    grid = np.round(np.arange(0.40, 0.52, 0.0005), 4)
    match = np.array([(np.round(u * k / 10) * 10 == c).mean() for k in grid])
    best_k = float(grid[match.argmax()])
    ls_k = float((c * u).sum() / (u * u).sum())  # 원점 통과 최소제곱 (검증용)

    by_type = {}
    for lt, g in df.groupby("load_type"):
        by_type[lt] = {
            "naive_ratio_kg_per_kWh": round(float(g["co2_kg"].sum() / g["Usage_kWh"].sum()), 4),
            "least_squares_kg_per_kWh": round(float((g["co2_kg"] * g["Usage_kWh"]).sum() / (g["Usage_kWh"] ** 2).sum()), 4),
            "mean_kW": round(float(g["kW"].mean()), 1),
            "p90_kW": round(float(g["kW"].quantile(0.9)), 1),
            "share_of_rows": round(float(len(g) / len(df)), 4),
        }
    return {
        "co2_factor_kg_per_kWh": best_k,
        "rounding_match_rate": round(float(match.max()), 4),
        "least_squares_factor_kg_per_kWh": round(ls_k, 4),
        "by_uci_load_type": by_type,
        "note": "UCI CO2는 사용전력 x 고정계수를 10kg 단위로 반올림한 값. "
                "Load_Type별 단순비율 차이는 저사용 구간의 반올림(0) 효과로 생긴 것으로, "
                "실질 계수는 Load_Type과 무관하게 거의 동일하다.",
    }


# ---------------------------------------------------------------------------
# 3. 요일 x 슬롯 프로파일
# ---------------------------------------------------------------------------
def build_weekly_profile(df):
    def mode(s):
        return s.value_counts().idxmax()

    g = df.groupby("slot_of_week")
    prof = pd.DataFrame({
        "base_load_kW": g["kW"].mean(),
        "base_load_median_kW": g["kW"].median(),
        "base_load_p90_kW": g["kW"].quantile(0.9),
        "uci_load_type": g["load_type"].agg(mode),
        "uci_maximum_share": g["load_type"].agg(lambda s: (s == "Maximum").mean()),
    }).reset_index()
    prof["weekday"] = prof["slot_of_week"] // SLOTS_PER_DAY
    prof["slot_of_day"] = prof["slot_of_week"] % SLOTS_PER_DAY
    prof["time_of_day"] = prof["slot_of_day"].map(lambda s: f"{s * SLOT_MIN // 60:02d}:{s * SLOT_MIN % 60:02d}")
    cols = ["slot_of_week", "weekday", "slot_of_day", "time_of_day", "base_load_kW",
            "base_load_median_kW", "base_load_p90_kW", "uci_load_type", "uci_maximum_share"]
    return prof[cols].round(3)


# ---------------------------------------------------------------------------
# 4. 계획 기간 energy_context 생성
# ---------------------------------------------------------------------------
def build_energy_context(tariff, weekly_profile, co2_factor, start, weeks, base_load_scale=1.0):
    tariff = tariff.copy()
    tariff["time_slot"] = pd.to_datetime(tariff["time_slot"])
    tariff["slot_of_week"] = (tariff["time_slot"].dt.dayofweek * SLOTS_PER_DAY
                              + (tariff["time_slot"].dt.hour * 60 + tariff["time_slot"].dt.minute) // SLOT_MIN)
    tou_week = tariff.drop_duplicates("slot_of_week").set_index("slot_of_week")[
        ["tariff_type", "price_per_kWh", "peak_flag"]]

    slots = pd.date_range(start, periods=weeks * 7 * SLOTS_PER_DAY, freq=f"{SLOT_MIN}min")
    sow = slots.dayofweek * SLOTS_PER_DAY + (slots.hour * 60 + slots.minute) // SLOT_MIN
    ctx = pd.DataFrame({"time_slot": slots, "slot_of_week": sow})
    ctx = ctx.join(tou_week, on="slot_of_week")
    ctx = ctx.join(weekly_profile.set_index("slot_of_week")[
        ["base_load_kW", "base_load_p90_kW", "uci_load_type"]], on="slot_of_week")
    ctx["base_load_kW"] = (ctx["base_load_kW"] * base_load_scale).round(3)
    ctx["base_load_p90_kW"] = (ctx["base_load_p90_kW"] * base_load_scale).round(3)
    ctx["co2_factor_kg_per_kWh"] = co2_factor
    ctx["in_original_tariff"] = ctx["time_slot"].isin(tariff["time_slot"]).astype(int)
    ctx["peak_flag"] = ctx["peak_flag"].astype(int)
    return ctx.drop(columns=["slot_of_week"])


def tariff_alignment(weekly_profile, tariff):
    """가상 TOU 요금 구분과 UCI Load_Type(실제 공장 부하구분)의 정합성."""
    t = tariff.copy()
    t["time_slot"] = pd.to_datetime(t["time_slot"])
    t["slot_of_week"] = (t["time_slot"].dt.dayofweek * SLOTS_PER_DAY
                         + (t["time_slot"].dt.hour * 60 + t["time_slot"].dt.minute) // SLOT_MIN)
    t = t.drop_duplicates("slot_of_week").merge(weekly_profile[["slot_of_week", "uci_load_type", "weekday"]], on="slot_of_week")
    t["tou_load"] = t["tariff_type"].map(TARIFF_TO_LOAD)
    weekday = t[t["weekday"] < 5]
    ct = pd.crosstab(t["tou_load"], t["uci_load_type"])
    return {
        "agreement_rate_all_days": round(float((t["tou_load"] == t["uci_load_type"]).mean()), 4),
        "agreement_rate_weekdays": round(float((weekday["tou_load"] == weekday["uci_load_type"]).mean()), 4),
        "crosstab_tou_vs_uci": {r: {c: int(ct.loc[r, c]) for c in ct.columns} for r in ct.index},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci-path", default="./uci_data/Steel_industry_data.csv")
    ap.add_argument("--data-dir", default="./virtual_data")
    ap.add_argument("--uci-out-dir", default="./uci_data")
    ap.add_argument("--weeks", type=int, default=6, help="energy_context가 덮을 주 수 (스케줄 makespan보다 길게)")
    ap.add_argument("--base-load-scale", type=float, default=1.0, help="UCI 기저부하 배율 (공장 규모 조정)")
    args = ap.parse_args()

    data_dir, uci_out = Path(args.data_dir), Path(args.uci_out_dir)
    uci_out.mkdir(parents=True, exist_ok=True)

    uci = load_uci(args.uci_path)
    tariff = pd.read_csv(data_dir / "tou_tariff.csv")
    start = pd.to_datetime(tariff["time_slot"]).min().normalize()

    ef = estimate_emission_factor(uci)
    weekly = build_weekly_profile(uci)
    ctx = build_energy_context(tariff, weekly, ef["co2_factor_kg_per_kWh"], start, args.weeks, args.base_load_scale)

    weekly.to_csv(uci_out / "uci_weekly_profile.csv", index=False, encoding="utf-8-sig")
    ctx.to_csv(data_dir / "energy_context.csv", index=False, encoding="utf-8-sig")
    ctx[["time_slot", "tariff_type", "price_per_kWh", "peak_flag"]].to_csv(
        data_dir / "tou_tariff_extended.csv", index=False, encoding="utf-8-sig")

    summary = {
        "uci_rows": int(len(uci)),
        "uci_period": [str(uci["slot_start"].min()), str(uci["slot_start"].max())],
        "uci_base_load_kW": {
            "mean": round(float(uci["kW"].mean()), 1),
            "max": round(float(uci["kW"].max()), 1),
            "weekday_daytime_mean": round(float(uci[(uci.weekday < 5) & uci.slot_start.dt.hour.between(8, 20)]["kW"].mean()), 1),
            "night_mean": round(float(uci[uci.slot_start.dt.hour < 8]["kW"].mean()), 1),
        },
        "emission_factor": ef,
        "tou_vs_uci_load_type": tariff_alignment(weekly, tariff),
        "energy_context": {
            "start": str(ctx["time_slot"].min()),
            "end_exclusive": str(ctx["time_slot"].max() + pd.Timedelta(minutes=SLOT_MIN)),
            "slots": int(len(ctx)),
            "base_load_scale": args.base_load_scale,
        },
    }
    with open(uci_out / "uci_integration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
