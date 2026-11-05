"""테스트 공통 준비: 작은 표본으로 모델을 한 번 학습해 임시 폴더에 저장한다."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.train import DEFAULT_CSV, train_all  # noqa: E402


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    if not DEFAULT_CSV.exists():
        pytest.skip("데이터 없음: python scripts/download_data.py 먼저 실행")
    output_dir = tmp_path_factory.mktemp("artifacts")
    metrics = train_all(DEFAULT_CSV, artifact_dir=output_dir, report_dir=output_dir,
                        use_mlflow=False, sample_rows=4000)
    return output_dir / "bundle.joblib", metrics
