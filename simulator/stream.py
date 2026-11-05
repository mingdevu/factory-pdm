"""AI4I 데이터 행을 PLC·IoT 센서처럼 일정 간격으로 수집 API에 보낸다.

실행 예:
  python -m simulator.stream --interval 0.5 --machines 5
  python -m simulator.stream --inject-fault 3   # 3번 설비에 과부하(토크↑, 회전수↓) 주입 (데모용)

원본 데이터에는 설비 번호가 없으므로 행을 순서대로 1..N번 설비에 돌아가며 배정한다.
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import httpx

from model.features import load_raw
from model.train import DEFAULT_CSV, split_data


def inject_overload(reading: dict, rng: random.Random) -> dict:
    """데모용 이상 주입: 토크를 높이고 회전수를 낮춰 과부하 패턴을 만든다 (주입된 값임을 표시)."""
    faulty = dict(reading)
    faulty["torque_nm"] = round(rng.uniform(66, 76), 1)
    faulty["rpm"] = round(rng.uniform(1140, 1230))
    faulty["tool_wear_min"] = max(reading["tool_wear_min"], rng.randint(205, 240))
    faulty["actual_failure"] = None  # 주입 데이터는 실제 라벨이 없다
    return faulty


def build_readings(machines: int, start: int, limit: int | None, test_only: bool = True):
    frame = load_raw(DEFAULT_CSV)
    if test_only:
        # 학습에 쓴 행을 흘리면 대시보드의 '예측 적중'이 부풀려지므로, 기본은 테스트셋 행만 보낸다
        _, _, test = split_data(frame)
        frame = frame.loc[sorted(test.index)]
    frame = frame.iloc[start:]
    if limit:
        frame = frame.iloc[:limit]
    for position, row in enumerate(frame.itertuples(index=False)):
        yield {
            "machine_id": position % machines + 1,
            "source_udi": int(row.UDI),
            "product_type": row.product_type,
            "air_temp_k": float(row.air_temp_k),
            "process_temp_k": float(row.process_temp_k),
            "rpm": float(row.rpm),
            "torque_nm": float(row.torque_nm),
            "tool_wear_min": float(row.tool_wear_min),
            "actual_failure": int(row.machine_failure),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-url", default="http://127.0.0.1:8001")
    parser.add_argument("--interval", type=float, default=1.0, help="전송 간격(초)")
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inject-fault", type=int, default=None, help="이상을 주입할 설비 번호")
    parser.add_argument("--all-rows", action="store_true", help="학습 행까지 모두 전송 (기본: 테스트셋만)")
    parser.add_argument("--loop", action="store_true", help="데이터를 다 보내면 처음부터 다시 보낸다 (상시 데모용)")
    args = parser.parse_args()

    rng = random.Random(0)
    sent = failed = 0
    with httpx.Client(base_url=args.ingest_url, timeout=5.0) as client:
        while True:
            for reading in build_readings(args.machines, args.start, args.limit, test_only=not args.all_rows):
                if args.inject_fault == reading["machine_id"]:
                    reading = inject_overload(reading, rng)
                try:
                    client.post("/readings", json=reading).raise_for_status()
                    sent += 1
                except httpx.HTTPError as error:
                    failed += 1
                    print(f"전송 실패 (machine {reading['machine_id']}): {error}", file=sys.stderr)
                if sent % 50 == 0 and sent:
                    print(f"전송 {sent}건 / 실패 {failed}건", flush=True)
                time.sleep(args.interval)
            if not args.loop:
                break
    print(f"완료: 전송 {sent}건 / 실패 {failed}건")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
