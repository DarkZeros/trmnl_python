import argparse
import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import os
import shutil
import struct
import zlib

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
        await page.screenshot(path=output, type='png', full_page=full_page)
        await browser.close()

def convert_to_bmp3_1bit(input_file: str, output_file: str):
    """Convert image to BMP v3, 1-bit monochrome."""
    img = Image.open(input_file)
    bw = img.convert("1")  # enforce 1-bit, black & white
    bw.save(output_file, format="BMP")  # Pillow defaults to BMP v3

def convert_to_png_2bit_gray(input_file: str, output_file: str):
    """Convert image to a 4-level (2-bit) grayscale PNG, Floyd-Steinberg dithered.

    This is the format the TRMNL firmware shows in its 4-gray mode. Pillow can't write 2-bit
    grayscale PNGs, so the file is assembled by hand (color type 0, bit depth 2).
    """
    levels = [0, 85, 170, 255]  # black, dark gray, light gray, white
    pal = Image.new("P", (1, 1))
    pal.putpalette([c for v in levels for c in (v, v, v)] + [0] * 3 * (256 - len(levels)))
    img = Image.open(input_file).convert("L").convert("RGB")
    indexed = img.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG)  # index == gray level 0..3

    w, h = indexed.size
    px = indexed.tobytes()
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter: none
        row = px[y * w:(y + 1) * w]
        for x in range(0, w, 4):
            b = 0
            for i in range(4):
                b = (b << 2) | (row[x + i] & 3 if x + i < w else 0)
            raw.append(b)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 2, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(output_file, "wb") as f:
        f.write(png)

def crop_and_pack(input_file: str, output_file: str, regions: list, canvas_size=(800, 480), bg_color="white"):
    """
    Crop specified regions and pack them into a fixed-size canvas.
    Regions are placed top-to-bottom, then left-to-right (like word wrapping).
    
    regions = [(x,y,w,h), ...]
    canvas_size = (width, height)
    """
    base_img = Image.open(input_file)
    canvas_w, canvas_h = canvas_size
    new_img = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    cropped_imgs = [base_img.crop((x, y, x+w, y+h)) for (x, y, w, h) in regions]

    x_offset, y_offset = 0, 0
    col_width = 0

    for img in cropped_imgs:
        w, h = img.size

        # If it doesn’t fit vertically, move to next column
        if y_offset + h > canvas_h:
            x_offset += col_width
            y_offset = 0
            col_width = 0

        # If it doesn’t fit horizontally, stop placing
        if x_offset + w > canvas_w:
            print("⚠️ Some cropped regions don’t fit in the canvas, skipped.")
            break

        # Paste image
        new_img.paste(img, (x_offset, y_offset))

        # Update offsets
        y_offset += h
        col_width = max(col_width, w)

    new_img.save(output_file)

def parse_crop_regions(region_strs):
    regions = []
    for r in region_strs:
        try:
            x, y, w, h = map(int, r.split(","))
            regions.append((x, y, w, h))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid crop region format: {r}. Use x,y,w,h")
    return regions

def main():
    parser = argparse.ArgumentParser(description="Take a website screenshot using Playwright.")
    parser.add_argument("url", help="HTTPS Website URL to capture")
    parser.add_argument("output", help="Output screenshot filename (e.g., screenshot.png)")
    parser.add_argument("--full-page", action="store_true", help="Capture the full page instead of just the viewport")
    parser.add_argument("--width", type=int, default=800, help="Viewport width (default: 800)")
    parser.add_argument("--height", type=int, default=480, help="Viewport height (default: 480)")
    parser.add_argument("--bmp3", action="store_true", help="Convert output to BMP v3 (1-bit black & white)")
    parser.add_argument("--gray4", action="store_true", help="Convert output to a 4-level (2-bit) grayscale PNG")
    parser.add_argument("--no-ignore-cert-errors", action="store_true", help="Do not ignore SSL/TLS certificate errors")
    parser.add_argument("--wait", type=int, default=0, help="Wait time in seconds before taking screenshot (default: 0)")
    parser.add_argument("--crop", nargs="+", help="Crop regions in x,y,w,h format (multiple allowed)")
    parser.add_argument("--canvas-width", type=int, default=800, help="Final canvas width")
    parser.add_argument("--canvas-height", type=int, default=480, help="Final canvas height")
    
    args = parser.parse_args()

    # Run Playwright screenshot
    asyncio.run(screenshot(
        args.url, args.output, args.full_page, args.width, args.height,
        not args.no_ignore_cert_errors, args.wait
    ))

    # Crop if requested
    if args.crop:
        regions = parse_crop_regions(args.crop)
        root, ext = os.path.splitext(args.output)
        shutil.copyfile(args.output, f"{root}_ori.{ext}")
        cropped_output = root + ext
        crop_and_pack(
            args.output, cropped_output, regions,
            canvas_size=(args.canvas_width, args.canvas_height)
        )
        args.output = cropped_output
        print(f"Cropped regions packed into: {cropped_output}")

    # Convert to BMP3 if requested
    if args.bmp3:
        root, _ = os.path.splitext(args.output)
        bmp_output = root + ".bmp"
        convert_to_bmp3_1bit(args.output, bmp_output)
        print(f"Converted to BMP v3 (1-bit): {bmp_output}")

    # Convert to 4-level grayscale PNG if requested (written next to the output as <root>.png)
    if args.gray4:
        root, _ = os.path.splitext(args.output)
        png_output = root + ".png"
        convert_to_png_2bit_gray(args.output, png_output)
        print(f"Converted to PNG (2-bit grayscale): {png_output}")

if __name__ == "__main__":
    main()
