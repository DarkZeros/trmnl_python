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


def pick_grid(n: int, canvas_w: int, canvas_h: int, gap: int) -> Tuple[int, int, int, int]:
    """
    Choose cols, rows to maximize tile area. Return (cols, rows, cell_w, cell_h).
    """
    best = None
    for cols in range(1, n + 1):
        rows = ceil(n / cols)
        # Total gaps
        total_gap_w = gap * (cols + 1)
        total_gap_h = gap * (rows + 1)
        cell_w = (canvas_w - total_gap_w) // cols
        cell_h = (canvas_h - total_gap_h) // rows
        if cell_w <= 0 or cell_h <= 0:
            continue
        area = cell_w * cell_h
        score = area  # could weight aspect ratio if desired
        if best is None or score > best[0]:
            best = (score, cols, rows, cell_w, cell_h)

    if best is None:
        raise ValueError("Canvas too small for any tiles with current gap")
    _, cols, rows, cell_w, cell_h = best
    return cols, rows, cell_w, cell_h


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
    header_h = max(16, min(40, cell_h // 10))
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


async def run(args):
    # Parse captures
    captures: List[CaptureSpec] = [parse_capture_spec(s) for s in args.capture]

    if len(captures) == 0:
        raise SystemExit("At least one --capture is required")

    canvas_w, canvas_h = parse_size(args.size)

    # Group captures by URL (to avoid reloading the same site)
    by_url: Dict[str, List[CaptureSpec]] = {}
    for c in captures:
        by_url.setdefault(c.url, []).append(c)

    # Decide viewport per URL: choose the largest requested (or default)
    def default_vp():
        return (1280, 1600)

    viewports_by_url: Dict[str, Tuple[int, int]] = {}
    waits_by_url: Dict[str, List[int]] = {}
    for url, lst in by_url.items():
        vp_list = [c.viewport for c in lst if c.viewport]
        if vp_list:
            max_w = max(v[0] for v in vp_list)
            max_h = max(v[1] for v in vp_list)
            viewports_by_url[url] = (max_w, max_h)
        else:
            viewports_by_url[url] = default_vp()
        waits_by_url[url] = [c.wait_ms for c in lst]

    # Take screenshots per unique URL
    screenshots: Dict[str, Image.Image] = {}
    async with async_playwright() as p:
        for url in by_url.keys():
            screenshots[url] = await capture_fullpage_png(
                p, args.browser, url, viewports_by_url[url], waits_by_url[url]
            )

    # Prepare layout
    n_tiles = len(captures)
    cols, rows, cell_w, cell_h = pick_grid(n_tiles, canvas_w, canvas_h, args.gap)

    # Build each tile
    tiles: List[Image.Image] = []
    for c in captures:
        full = screenshots[c.url]
        # Clip region to page bounds
        x = max(0, c.x)
        y = max(0, c.y)
        w = max(0, min(c.w, full.width - x))
        h = max(0, min(c.h, full.height - y))
        if w <= 0 or h <= 0:
            # Create an empty tile with an error header
            region = Image.new("RGB", (1, 1), color=(255, 255, 255))
            label = (c.name or f"{hostname(c.url)}") + " [invalid region]"
        else:
            region = full.crop((x, y, x + w, y + h))
            label = c.name or f"{hostname(c.url)}  {c.x},{c.y},{c.w},{c.h}"

        tile = make_tile(
            region_img=region,
            label=label,
            cell_w=cell_w,
            cell_h=cell_h,
            padding=args.tile_padding,
            draw_borders=(not args.no_borders),
        )
        tiles.append(tile)

    # Compose onto final canvas (grayscale first for best 1-bit conversion)
    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    # place tiles with gaps
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_tiles:
                break
            x = args.gap + c * (cell_w + args.gap)
            y = args.gap + r * (cell_h + args.gap)
            canvas.paste(tiles[idx], (x, y))
            idx += 1

    # Convert to 1-bit and save as BMP (BMP v3 1bpp)
    out_1b = canvas.convert("1")
    # Pillow writes BMP v3 by default for 1-bit images
    out_1b.save(args.output, format="BMP")

    print(
        f"Saved {n_tiles} tile(s) to {args.output} "
        f"as {canvas_w}x{canvas_h} BMP (1-bit)."
    )


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
        default=4,
        help="Gap/padding between tiles on the final canvas (default: 4)",
    )
    p.add_argument(
        "--tile-padding",
        type=int,
        default=6,
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
