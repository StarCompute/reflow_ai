"""把 slides.html 里的每一张 .slide 导出为 1080x1920 的 PNG（抖音竖版）。

用法：
    cd proctune/docs/douyin
    python export_images.py
导出结果： img/01.png ... img/16.png（按幻灯片在 HTML 中的顺序）

依赖： pip install playwright && playwright install chromium
"""
import pathlib

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent
HTML = (HERE / "slides.html").as_uri()
OUT = HERE / "img"
OUT.mkdir(exist_ok=True)

SCALE = 2  # 2x 清晰度：实际输出 2160x3840，上传抖音更清晰；设 1 则为 1080x1920


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=SCALE)
        page.goto(HTML)
        page.wait_for_timeout(400)  # 等字体/布局稳定
        slides = page.query_selector_all(".slide")
        for i, s in enumerate(slides, 1):
            path = OUT / f"{i:02d}.png"
            s.screenshot(path=str(path))
            print(f"  -> {path.name}")
        browser.close()
    print(f"完成：共导出 {len(slides)} 张图片到 {OUT}")


if __name__ == "__main__":
    main()
