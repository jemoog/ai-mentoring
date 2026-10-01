"""
전체 실험 파이프라인 한 번에 실행.

순서
1) UCI 데이터 결합      : uci_energy_integration.py  → virtual_data/energy_context.csv 등 생성
2) 기준 스케줄러 4종    : FIFO, EDD, Energy-aware v1, Slack-aware v1
3) 개선 스케줄러        : Slack-aware v2 (UCI 결합 + 납기 보호 + 작업 순서 탐색)
4) 공통 지표 비교       : compare_schedules.py  → common_schedule_comparison.csv, improvement_vs_edd.csv
5) 제약조건 검증        : validate_schedule.py  → constraint_validation.csv

실행 (프로젝트 루트에서):
    python src/run_all.py
    python src/run_all.py --iterations 300      # v2 작업순서 탐색 반복 수 (기본 1500, 약 2분)
    python src/run_all.py --skip-baselines      # v2와 비교/검증만 다시 실행
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def run(args, label):
    t = time.time()
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([sys.executable, *map(str, args)], cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"[실패] {label} (exit {r.returncode})")
    print(f"--- {label} 완료 ({time.time() - t:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "virtual_data"))
    ap.add_argument("--uci-path", default=str(ROOT / "uci_data" / "Steel_industry_data.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "output_uci"))
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--params", default=None, help="v2 파라미터 JSON (선택)")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run(["uci_energy_integration.py", "--uci-path", args.uci_path, "--data-dir", data,
         "--uci-out-dir", ROOT / "uci_data"], "1. UCI 데이터 결합")
    if not args.skip_baselines:
        for f, label in [("fifo_scheduler.py", "FIFO"), ("edd_scheduler.py", "EDD"),
                         ("energy_aware_scheduler.py", "Energy-aware v1"),
                         ("slack_aware_energy_scheduler.py", "Slack-aware v1")]:
            run([f, "--input-dir", data, "--output-dir", out], f"2. {label}")
    v2_args = ["slack_aware_energy_scheduler_v2.py", "--input-dir", data, "--output-dir", out,
               "--iterations", args.iterations]
    if args.params:
        v2_args += ["--params", args.params]
    run(v2_args, "3. Slack-aware v2")
    run(["compare_schedules.py", "--data-dir", data, "--schedule-dir", out, "--output-dir", out], "4. 공통 지표 비교")
    run(["validate_schedule.py", "--data-dir", data, "--schedule-dir", out], "5. 제약조건 검증")
    print(f"\n결과: {out / 'common_schedule_comparison.csv'}")
    print(f"      {out / 'improvement_vs_edd.csv'}")


if __name__ == "__main__":
    main()
