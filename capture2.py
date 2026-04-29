#!/usr/bin/env python3
import argparse
import asyncio
from dataclasses import dataclass
from io import BytesIO
from math import ceil
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright


@dataclass
class CaptureSpec:
    url: str
    x: int
    y: int
    w: int
    h: int
    name: Optional[str] = None
    wait_ms: int = 0
    viewport: Optional[Tuple[int, int]] = None  # (width, height)


def parse_size(s: str) -> Tuple[int, int]:
    try:
        w_str, h_str = s.lower().split("x")
        return int(w_str), int(h_str)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid size '{s}', expected WIDTHxHEIGHT")


def parse_capture_spec(spec: str) -> CaptureSpec:
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    kv: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            raise argparse.ArgumentTypeError(
                f"Invalid capture part '{p}', expected key=value"
            )
        k, v = p.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    required = ["url", "x", "y", "w", "h"]
    for r in required:
        if r not in kv:
            raise argparse.ArgumentTypeError(
                f"Missing '{r}' in capture spec: {spec}"
            )

    def to_int(key: str) -> int:
        try:
            return int(kv[key])
        except Exception:
            raise argparse.ArgumentTypeError(f"'{key}' must be an integer in: {spec}")

    viewport: Optional[Tuple[int, int]] = None
    if "viewport" in kv:
        viewport = parse_size(kv["viewport"])

    wait_ms = int(kv.get("wait", kv.get("wait_ms", "0")))

    return CaptureSpec(
        url=kv["url"],
        x=to_int("x"),
        y=to_int("y"),
        w=to_int("w"),
        h=to_int("h"),
        name=kv.get("name"),
        wait_ms=wait_ms,
        viewport=viewport,
    )


def fit_inside(src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[int, int]:
    if src_w == 0 or src_h == 0:
        return 0, 0
    scale = min(dst_w / src_w, dst_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))


def hostname(u: str) -> str:
    try:
        return urlparse(u).hostname or u
    except Exception:
        return u


def draw_header(draw: ImageDraw.ImageDraw, rect: Tuple[int, int, int, int], label: str, font: ImageFont.ImageFont):
    x0, y0, x1, y1 = rect
    # Black header bar
    draw.rectangle(rect, fill=0)
    # Measure and truncate text to fit
    max_w = x1 - x0 - 8  # 4px padding either side
    text = label
    # If font has getlength (Pillow 10+), use it; else textbbox
    def text_width(s: str) -> int:
        try:
            return int(font.getlength(s))
        except Exception:
            return draw.textbbox((0,0), s, font=font)[2]

    ellipsis = "…"
    while text_width(text) > max_w and len(text) > 1:
        # Keep a bit, add ellipsis
        cut = max(1, len(text) - 2)
        text = text[:cut] + ellipsis
    tw = text_width(text)
    th = draw.textbbox((0,0), text, font=font)[3]
    tx = x0 + 4
    ty = y0 + max(0, ((y1 - y0) - th) // 2)
    # White text on black bar
    draw.text((tx, ty), text, font=font, fill=255)


def make_tile(
    region_img: Image.Image,
    label: str,
    cell_w: int,
    cell_h: int,
    padding: int,
    draw_borders: bool,
) -> Image.Image:
    # Base tile (white)
    tile = Image.new("L", (cell_w, cell_h), color=255)  # start grayscale for crisp conversion
    draw = ImageDraw.Draw(tile)

    # Header height heuristic
    header_h = 16
    # Font selection: default bitmap font; if too tall, it will still fit
    font = ImageFont.load_default()

    # Header
    draw_header(draw, (0, 0, cell_w, header_h), label, font)

    # Available area for the image content
    avail_w = max(1, cell_w - 2 * padding)
    avail_h = max(1, cell_h - header_h - padding - padding)  # pad bottom too

    # Fit region into available box
    target_w, target_h = fit_inside(region_img.width, region_img.height, avail_w, avail_h)
    # Convert region to L for consistent 1-bit conversion later
    region_L = region_img.convert("L").resize((target_w, target_h), Image.LANCZOS)

    # Position (centered)
    x = (cell_w - target_w) // 2
    y = header_h + padding + (avail_h - target_h) // 2

    # Paste region (as grayscale)
    tile.paste(region_L, (x, y))

    # Optional border
    if draw_borders:
        draw.rectangle((0, 0, cell_w - 1, cell_h - 1), outline=0, width=1)

    return tile


async def capture_fullpage_png(
    playwright, browser_name: str, url: str, viewport: Tuple[int, int], waits: List[int]
) -> Image.Image:
    browser_launcher = {
        "chromium": playwright.chromium,
        "firefox": playwright.firefox,
        "webkit": playwright.webkit,
    }[browser_name]

    browser = await browser_launcher.launch(headless=True)
    context = await browser.new_context(
        viewport={"width": viewport[0], "height": viewport[1]},
        device_scale_factor=1.0,
        ignore_https_errors=True
    )
    page = await context.new_page()

    # Be conservative with timeout; user-provided waits are extra
    goto_timeout = 45000
    await page.goto(url, timeout=goto_timeout)

    # Apply the *maximum* requested extra wait among all regions for this URL
    extra_wait = max([0] + waits)
    if extra_wait > 0:
        await page.wait_for_timeout(extra_wait)

    png_bytes = await page.screenshot(full_page=True, type="png")
    await context.close()
    await browser.close()

    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    return img


@dataclass
class TileCrop:
    img: Image.Image
    label: str
    orig_w: int
    orig_h: int


def pack_rows(tiles: List[TileCrop], canvas_w: int, gap: int) -> List[List[TileCrop]]:
    """Pack tiles sequentially into rows based on natural width."""
    rows: List[List[TileCrop]] = []
    current: List[TileCrop] = []
    used_w = 0
    for t in tiles:
        w = t.orig_w + (gap if current else 0)
        if current and used_w + w > canvas_w:
            rows.append(current)
            current = [t]
            used_w = t.orig_w
        else:
            current.append(t)
            used_w += w
    if current:
        rows.append(current)
    return rows


async def run(args):
    captures: List[CaptureSpec] = [parse_capture_spec(s) for s in args.capture]
    if not captures:
        raise SystemExit("At least one --capture is required")
    canvas_w, canvas_h = parse_size(args.size)

    # group + screenshot
    by_url: Dict[str, List[CaptureSpec]] = {}
    for c in captures:
        by_url.setdefault(c.url, []).append(c)

    viewports_by_url: Dict[str, Tuple[int, int]] = {}
    waits_by_url: Dict[str, List[int]] = {}
    for url, lst in by_url.items():
        vp_list = [c.viewport for c in lst if c.viewport]
        if vp_list:
            max_w = max(v[0] for v in vp_list)
            max_h = max(v[1] for v in vp_list)
            viewports_by_url[url] = (max_w, max_h)
        else:
            viewports_by_url[url] = (1280, 1600)
        waits_by_url[url] = [c.wait_ms for c in lst]

    screenshots: Dict[str, Image.Image] = {}
    async with async_playwright() as p:
        for url in by_url:
            screenshots[url] = await capture_fullpage_png(
                p, args.browser, url, viewports_by_url[url], waits_by_url[url]
            )

    # Collect crops
    crops: List[TileCrop] = []
    for c in captures:
        full = screenshots[c.url]
        x, y = max(0, c.x), max(0, c.y)
        w = max(0, min(c.w, full.width - x))
        h = max(0, min(c.h, full.height - y))
        if w <= 0 or h <= 0:
            region = Image.new("RGB", (1, 1), "white")
            label = (c.name or hostname(c.url)) + " [invalid region]"
        else:
            region = full.crop((x, y, x + w, y + h))
            label = c.name or f"{hostname(c.url)} {c.x},{c.y},{c.w},{c.h}"
        crops.append(TileCrop(region, label, region.width, region.height))

    # Row packing (unscaled)
    rows = pack_rows(crops, canvas_w, args.gap)
    row_widths = [sum(t.orig_w for t in row) + args.gap * (len(row) + 1) for row in rows]
    row_heights = [max(t.orig_h for t in row) + args.tile_padding * 2 + 20 for row in rows]
    natural_w = max(row_widths)
    natural_h = sum(row_heights) + args.gap * (len(rows) + 1)

    # Global scale factor
    scale = min(canvas_w / natural_w, canvas_h / natural_h, 1.0)

    # Compose final canvas
    canvas = Image.new("L", (canvas_w, canvas_h), 255)
    y = args.gap
    for r_i, row in enumerate(rows):
        x = args.gap
        row_h = max(int(t.orig_h * scale) for t in row) + args.tile_padding * 2 + 20
        for t in row:
            tw = int(t.orig_w * scale)
            th = int(t.orig_h * scale)
            # Render tile now at final size
            tile_img = make_tile(
                t.img.resize((tw, th), Image.LANCZOS),
                t.label,
                tw + args.tile_padding * 2,
                th + args.tile_padding * 2 + 20,
                padding=args.tile_padding,
                draw_borders=(not args.no_borders),
            )
            canvas.paste(tile_img, (x, y))
            x += tile_img.width + args.gap
        y += row_h + args.gap

    out_1b = canvas.convert("1")
    out_1b.save(args.output, format="BMP")
    print(f"Saved {len(crops)} tile(s) to {args.output} with row-packer scaling.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Capture website regions into a tiled 1-bit BMP using Playwright + Pillow."
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output BMP path, e.g. out.bmp",
    )
    p.add_argument(
        "--size",
        default="800x480",
        help="Final canvas size, WIDTHxHEIGHT (default: 800x480)",
    )
    p.add_argument(
        "--capture",
        action="append",
        required=True,
        help=(
            "Capture spec (repeatable). Format: "
            "url=...;x=INT;y=INT;w=INT;h=INT;[name=STR];[wait=MS];[viewport=WxH]"
        ),
    )
    p.add_argument(
        "--gap",
        type=int,
        default=2,
        help="Gap/padding between tiles on the final canvas (default: 4)",
    )
    p.add_argument(
        "--tile-padding",
        type=int,
        default=2,
        help="Inner padding around each region inside its tile (default: 6)",
    )
    p.add_argument(
        "--no-borders",
        action="store_true",
        help="Do not draw 1px borders around tiles",
    )
    p.add_argument(
        "--browser",
        choices=["chromium", "firefox", "webkit"],
        default="chromium",
        help="Playwright browser engine (default: chromium)",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
