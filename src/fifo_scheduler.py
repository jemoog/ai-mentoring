"""
FIFO baseline scheduler.
- 작업을 release_time 빠른 순(동일 시 job_id 순)으로 처리
- 납기, 전력요금, 피크 시간대는 고려하지 않음
- 제약조건/후보 선택(가장 빨리 끝나는 설비·시간)은 EDD 스케줄러와 동일한 로직 재사용
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from edd_scheduler import build_edd_schedule, compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./virtual_data')
    ap.add_argument('--output-dir', default='./output')
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    job_orders = pd.read_csv(inp / 'job_orders.csv')
    machines = pd.read_csv(inp / 'machines.csv')
    maint = pd.read_csv(inp / 'maintenance_schedule.csv')
    routing = pd.read_csv(inp / 'product_routing.csv')
    schedule = build_edd_schedule(job_orders, machines, routing, maint, sort_cols=['release_time', 'job_id'])
    schedule.to_csv(out / 'fifo_schedule.csv', index=False, encoding='utf-8-sig')
    metrics = compute_metrics(schedule, job_orders.assign(due_time=pd.to_datetime(job_orders.due_time)))
    with open(out / 'fifo_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
