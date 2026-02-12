import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 上三级
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code_bench.tools.tools_rapidapi_clawer import RapidAPIFinanceScraper


logger = logging.getLogger(__name__)


def load_home_urls(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"home_url.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "home_url" in data:
        data = data["home_url"]
    if not isinstance(data, list):
        raise ValueError("home_url.json must be a JSON array of URLs")
    urls = [u for u in data if isinstance(u, str) and u.strip()]
    # de-dup preserve order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def normalize_home_url(url: str) -> str:
    url = url.strip()
    if not url.endswith("/"):
        url += "/"
    return url


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Subscribe RapidAPI from home_url.json")
    parser.add_argument(
        "--home_url_json",
        default="tools/home_url.json",
        help="JSON array of RapidAPI home URLs",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--output_dir", default="./tools/rapidapi/clawer")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    urls = load_home_urls(Path(args.home_url_json))
    urls = [normalize_home_url(u) for u in urls]

    scraper = RapidAPIFinanceScraper(
        tool_url_path="",
        headless=args.headless,
        output_dir=args.output_dir,
    )

    if not (os.path.exists(scraper.cookies_txt) or os.path.exists(scraper.cookies_pkl)):
        logger.info("⚠️ 未检测到 cookies")
        choice = input("是否现在登录？(y/n): ").strip().lower()
        if choice == "y":
            scraper.manual_login_and_save_cookies()
        else:
            logger.info("❌ 退出")
            return

    scraper._init_driver()
    if not scraper.is_logged_in():
        logger.info("❌ 登录失败")
        return

    with tqdm(total=len(urls), desc="Subscribe", unit="api", ncols=100) as pbar:
        for url in urls:
            api_name = url.split("/")[-2] if url.endswith("/") else url.split("/")[-1]
            pbar.set_description(f"Subscribe {api_name[:20]}")
            try:
                scraper.driver.get(url)
                time.sleep(2)
                if scraper._check_page_not_found():
                    logger.warning("❌ 页面不存在，跳过: %s", url)
                    pbar.update(1)
                    continue

                pricing_url = url + "pricing"
                scraper.subscribe_to_api(pricing_url, pbar, max_retries=args.max_retries)
            except Exception as e:
                logger.error("❌ 订阅失败: %s error=%s", url, e)
            pbar.update(1)
            time.sleep(args.sleep)

    scraper._quit_driver()


if __name__ == "__main__":
    main()
