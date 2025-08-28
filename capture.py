import argparse
import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import os

async def screenshot(url: str, output: str, full_page: bool, width: int, height: int, ignore_cert_errors: bool, wait: int):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            ignore_https_errors=ignore_cert_errors
        )
        page = await context.new_page()
        await page.goto(url)

        # Wait if requested
        if wait > 0:
            await page.wait_for_timeout(wait * 1000)

        # Always screenshot first (in original format)
        await page.screenshot(path=output, full_page=full_page)
        await browser.close()

def convert_to_bmp3_1bit(input_file: str, output_file: str):
    """Convert image to BMP v3, 1-bit monochrome (like `convert -monochrome -colors 2 -depth 1 -strip bmp3:`)."""
    img = Image.open(input_file)
    bw = img.convert("1")  # enforce 1-bit, black & white
    bw.save(output_file, format="BMP")  # Pillow defaults to BMP v3

def main():
    parser = argparse.ArgumentParser(description="Take a website screenshot using Playwright.")
    parser.add_argument("url", help="HTTPS Website URL to capture")
    parser.add_argument("output", help="Output screenshot filename (e.g., screenshot.png)")
    parser.add_argument("--full-page", action="store_true", help="Capture the full page instead of just the viewport")
    parser.add_argument("--width", type=int, default=800, help="Viewport width (default: 800)")
    parser.add_argument("--height", type=int, default=480, help="Viewport height (default: 480)")
    parser.add_argument("--bmp3", action="store_true", help="Convert output to BMP v3 (1-bit black & white)")
    parser.add_argument("--no-ignore-cert-errors", action="store_true", help="Do not ignore SSL/TLS certificate errors")
    parser.add_argument("--wait", type=int, default=0, help="Wait time in seconds before taking screenshot (default: 0)")
    
    args = parser.parse_args()

    # Run Playwright screenshot
    asyncio.run(screenshot(
        args.url, args.output, args.full_page, args.width, args.height,
        not args.no_ignore_cert_errors, args.wait
    ))

    # Convert to BMP3 if requested
    if args.bmp3:
        root, _ = os.path.splitext(args.output)
        bmp_output = root + ".bmp"
        convert_to_bmp3_1bit(args.output, bmp_output)
        print(f"Converted to BMP v3 (1-bit): {bmp_output}")

if __name__ == "__main__":
    main()

