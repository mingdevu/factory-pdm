"""UCI AI4I 2020 Predictive Maintenance Dataset을 data/raw/에 내려받는다."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "ai4i2020.csv"
    if target.exists():
        print(f"이미 존재: {target}")
        return 0
    try:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
    except Exception as error:  # 네트워크 실패는 숨기지 않고 알린다
        print(f"다운로드 실패: {error}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extract("ai4i2020.csv", RAW_DIR)
    print(f"저장 완료: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
