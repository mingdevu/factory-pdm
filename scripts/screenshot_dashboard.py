"""Playwright로 Streamlit 대시보드 스크린샷을 찍는다 (README용).

사용: python scripts/screenshot_dashboard.py --url http://127.0.0.1:8501 --out docs/dashboard.png
(개발용 도구라 requirements.txt에는 넣지 않았다: pip install playwright && playwright install chromium)
"""
import argparse

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--out", default="docs/dashboard.png")
    parser.add_argument("--wait-text", default="설비별 최신 상태")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1100}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text(args.wait_text).first.wait_for(timeout=60_000)
        page.wait_for_timeout(4000)  # 차트 렌더링 대기
        page.screenshot(path=args.out, full_page=True)
        browser.close()
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
