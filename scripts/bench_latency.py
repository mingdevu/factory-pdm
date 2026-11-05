"""실행 중인 serve/ingest API의 응답 지연을 측정한다 (README 수치의 근거).

사용: python scripts/bench_latency.py --n 200
주의: 측정 결과는 실행한 PC 사양·부하에 따라 달라진다.
"""
import argparse
import json
import statistics
import time

import httpx

READING = {"machine_id": 1, "product_type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0,
           "rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 20}


def measure(client: httpx.Client, url: str, n: int) -> dict:
    client.post(url, json=READING).raise_for_status()  # 워밍업 1회
    samples = []
    for _ in range(n):
        started = time.perf_counter()
        client.post(url, json=READING).raise_for_status()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {"n": n, "p50_ms": round(statistics.median(samples), 1),
            "p95_ms": round(samples[int(n * 0.95) - 1], 1), "max_ms": round(samples[-1], 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--serve-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ingest-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    with httpx.Client(timeout=10) as client:
        results = {
            "serve /predict": measure(client, f"{args.serve_url}/predict", args.n),
            "serve /anomaly": measure(client, f"{args.serve_url}/anomaly", args.n),
            "ingest /readings (저장+예측+이상점수)": measure(client, f"{args.ingest_url}/readings", args.n),
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
