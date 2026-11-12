import os
import uuid
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, FileResponse

# Configure logger (will be set up later with argparse)
logger = logging.getLogger(__name__)

app = FastAPI()

# Paths for persistent storage
DATA_FOLDER = "data"
DEVICES_FILE = os.path.join(DATA_FOLDER, "devices.json")
os.makedirs(DATA_FOLDER, exist_ok=True)

# In-memory storage for devices and logs
devices = {}
logs = []
address = ""

# Global server settings
SETTINGS = {
    # OTA: when "firmware_upgrading" is on and firmware/<firmware_file> exists, any device that
    # reports a FW-Version other than "firmware_version" is told to download and install it.
    # The device then reports the new version and the offer stops by itself, so the version
    # compiled into the .bin (FW_*_VERSION in the firmware's include/config.h) must match.
    "firmware_upgrading": False,
    "firmware_version": "1.8.17",
    "firmware_file": "firmware.bin",
    "firmware_download_url": None,  # None: serve firmware/<firmware_file> from this server
    "default_image": "default.bmp",
    "default_refresh_rate": "1800",
    "pre_refresh": 20,
    # Slower refresh at night: set "night_refresh_rate" (seconds) plus "night_start"/"night_end"
    # ("HH:MM", local time; the window may wrap midnight). Overridable per device.
    "night_refresh_rate": None,
    "night_start": "23:00",
    "night_end": "07:00",
}

# Optional overrides for SETTINGS, e.g. {"firmware_upgrading": true}. Re-read whenever the file
# changes, and at least once an hour regardless, so OTA can be switched on and off without
# touching the code or restarting.
SETTINGS_FILE = os.path.join(DATA_FOLDER, "settings.json")
SETTINGS_REREAD_SECONDS = 3600
DEFAULT_SETTINGS = dict(SETTINGS)
_settings_mtime = None
_settings_read_at = None

def load_settings_overrides():
    global _settings_mtime, _settings_read_at
    try:
        mtime = os.path.getmtime(SETTINGS_FILE)
    except OSError:
        mtime = None
    now = time.monotonic()
    fresh = _settings_read_at is not None and now - _settings_read_at < SETTINGS_REREAD_SECONDS
    if mtime == _settings_mtime and fresh:
        return
    _settings_mtime = mtime
    _settings_read_at = now
    overrides = {}
    if mtime is not None:
        try:
            with open(SETTINGS_FILE) as f:
                overrides = json.load(f)
            if not isinstance(overrides, dict):
                raise ValueError("expected a JSON object")
        except (OSError, ValueError) as e:
            logger.error(f"Ignoring {SETTINGS_FILE}: {e}")
            overrides = {}
    unknown = [k for k in overrides if k not in DEFAULT_SETTINGS]
    if unknown:
        logger.warning(f"Unknown keys in {SETTINGS_FILE}: {unknown}")
    SETTINGS.clear()
    SETTINGS.update(DEFAULT_SETTINGS)
    SETTINGS.update({k: v for k, v in overrides.items() if k in DEFAULT_SETTINGS})
    applied = {k: v for k, v in SETTINGS.items() if v != DEFAULT_SETTINGS[k]}
    if applied != getattr(load_settings_overrides, "last_applied", None):
        logger.info(f"Settings overrides from {SETTINGS_FILE}: {applied}")
    load_settings_overrides.last_applied = applied

# Ensure images folder exists
IMAGE_FOLDER = "images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# OTA firmware images (the PlatformIO build's firmware.bin) go here
FIRMWARE_FOLDER = "firmware"
os.makedirs(FIRMWARE_FOLDER, exist_ok=True)


def load_devices():
    global devices
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, "r") as f:
                devices = json.load(f)
            logger.info(f"Loaded {len(devices)} devices from {DEVICES_FILE}")
        except Exception as e:
            logger.error(f"Failed to load devices: {e}")
            devices = {}
    else:
        devices = {}


def save_devices():
    try:
        with open(DEVICES_FILE, "w") as f:
            json.dump(devices, f, indent=2)
        logger.debug("Devices saved to disk")
    except Exception as e:
        logger.error(f"Failed to save devices: {e}")


def device_setting(device, key):
    return device.get(key, SETTINGS[key])


def effective_refresh_rate(device, now=None):
    """Refresh rate for this wake, in seconds: the night rate inside the night window.

    The night sleep is cut short at the window's end so the device is back on its day rate
    on time instead of up to one night interval late.
    """
    rate = int(device.get("refresh_rate", SETTINGS["default_refresh_rate"]))
    night_rate = device_setting(device, "night_refresh_rate")
    if not night_rate:
        return rate
    now = now or datetime.now()
    start = datetime.strptime(device_setting(device, "night_start"), "%H:%M").time()
    end = datetime.strptime(device_setting(device, "night_end"), "%H:%M").time()
    t = now.time()
    in_night = (start <= t < end) if start <= end else (t >= start or t < end)
    if not in_night:
        return rate
    end_dt = datetime.combine(now.date(), end)
    if end_dt <= now:
        end_dt += timedelta(days=1)
    until_end = int((end_dt - now).total_seconds())
    return max(rate, min(int(night_rate), until_end))


# Accept both /api/setup and /api/setup/
@app.get("/api/setup", include_in_schema=True)
@app.get("/api/setup/", include_in_schema=False)
async def setup(ID: str = Header(None)):
    logger.debug(f"/api/setup called with ID={ID}")

    if not ID:
        logger.warning("/api/setup missing ID header")
        return JSONResponse(status_code=400, content={"status": 400, "error": "Missing ID header"})

    # If device not registered, create new entry
    if ID not in devices:
        api_key = hashlib.sha1(ID.encode()).hexdigest()[:22]
        friendly_id = uuid.uuid4().hex[:6].upper()
        devices[ID] = {
            "api_key": api_key,
            "friendly_id": friendly_id,
            "image": SETTINGS["default_image"],
            "refresh_rate": SETTINGS["default_refresh_rate"],
            "pre_refresh": SETTINGS["pre_refresh"],
        }
        logger.info(f"New device registered: {devices[ID]}")
        save_devices()

    device = devices[ID]
    response = {
        "status": 200,
        "api_key": device["api_key"],
        "friendly_id": device["friendly_id"],
        "image_url": f"/static/{device['image']}",
        "filename": "empty_state",
    }
    logger.debug(f"/api/setup response: {response}")
    return response


import asyncio

# At most one pending update per device; otherwise every /api/display call (retries included)
# queues another capture.
pending_updates = {}

# One capture at a time: concurrent headless Chromiums starve each other on the server and
# page.screenshot() times out. (Captures used to be serialized by blocking the event loop.)
capture_lock = asyncio.Lock()

async def run_update_script_later(device, delay):
    await asyncio.sleep(delay)
    script_path = os.path.join("scripts", f"{device['friendly_id']}.sh")
    if os.path.exists(script_path):
        try:
            logger.info(f"Running update script for {device['friendly_id']} after {delay}s")
            # "image_format": "png" makes the script render a 4-gray PNG instead of a 1-bit BMP
            new_image = f"{device['friendly_id']}.{device.get('image_format', 'bmp')}"
            # In a worker thread: a browser capture takes seconds and would otherwise block the
            # event loop, so the device's own API calls time out while it runs.
            # Render into a temp file and swap it in when complete: capture scripts write
            # intermediate images (raw screenshot, crop) to their output path, and a device waking
            # mid-capture would otherwise download and show one of those.
            tmp_path = os.path.join(IMAGE_FOLDER, f".tmp_{new_image}")
            async with capture_lock:
                await asyncio.to_thread(subprocess.run, [script_path, tmp_path], check=True, timeout=180)
            os.replace(tmp_path, os.path.join(IMAGE_FOLDER, new_image))
            device["image"] = new_image
            save_devices()
            logger.info(f"Updated image for {device['friendly_id']} to {new_image}")
        except Exception as e:
            logger.error(f"Error running script for {device['friendly_id']}: {e}")

@app.get("/api/display")
async def display(
    ID: str = Header(None),
    Access_Token: str = Header(None, alias="Access-Token"),
    Refresh_Rate: str = Header(None, alias="Refresh-Rate"),
    Battery_Voltage: str = Header(None, alias="Battery-Voltage"),
    FW_Version: str = Header(None, alias="FW-Version"),
    RSSI: str = Header(None),
    Wake_Time: str = Header(None, alias="Wake-Time"),
    Timings: str = Header(None),
):
    logger.debug(f"/api/display called with headers: ID={ID}, Access_Token={Access_Token}, Refresh_Rate={Refresh_Rate}, Battery_Voltage={Battery_Voltage}, FW_Version={FW_Version}, RSSI={RSSI}")
    if Timings:
        # Previous wake cycle, per phase in ms (sent by the instrumented firmware)
        logger.info(f"Timings from {ID}: {Timings} (battery={Battery_Voltage}V, rssi={RSSI})")
    elif Wake_Time:
        logger.info(f"Wake time from {ID}: {Wake_Time} ms (battery={Battery_Voltage}V, rssi={RSSI})")

    if not ID or not Access_Token:
        logger.warning("/api/display missing required headers")
        return JSONResponse(status_code=400, content={"status": 400, "error": "Missing headers"})

    load_settings_overrides()

    device = devices.get(ID)
    if not device or device["api_key"] != Access_Token:
        logger.error(f"Device not found or token mismatch for ID={ID}")
        return {"status": 500, "error": "Device not found"}
        
    try:
        refresh_rate = effective_refresh_rate(device)
    except ValueError as e:
        logger.error(f"Bad refresh/night settings for {device['friendly_id']}: {e}")
        refresh_rate = int(device.get("refresh_rate", SETTINGS["default_refresh_rate"]))

    # Update the image dynamically if a script exists
    # Schedule update script execution (refresh_rate - 20s)
    try:
        refresh_delay = refresh_rate - int(device_setting(device, "pre_refresh"))
        task = pending_updates.get(ID)
        if refresh_delay > 0 and (task is None or task.done()):
            pending_updates[ID] = asyncio.create_task(run_update_script_later(device, refresh_delay))
    except Exception as e:
        logger.error(f"Failed to schedule script execution for {device['friendly_id']}: {e}")

    # Determine firmware update requirement
    update_firmware = bool(SETTINGS["firmware_upgrading"]) and FW_Version != SETTINGS["firmware_version"]
    firmware_url = SETTINGS["firmware_download_url"]
    if update_firmware and not firmware_url:
        if os.path.exists(os.path.join(FIRMWARE_FOLDER, SETTINGS["firmware_file"])):
            firmware_url = f"{address}/firmware/{SETTINGS['firmware_file']}"
        else:
            logger.warning(f"Firmware upgrade is on but {FIRMWARE_FOLDER}/{SETTINGS['firmware_file']} is missing")
            update_firmware = False
    if update_firmware:
        logger.info(f"Offering firmware {SETTINGS['firmware_version']} to {device['friendly_id']} (has {FW_Version}): {firmware_url}")

    # The device skips the download and the e-paper refresh when the filename matches what it is
    # already showing, so name the image by its content: re-rendering an identical dashboard
    # then costs the device nothing.
    file_path = os.path.join(IMAGE_FOLDER, device["image"])
    filename = "empty"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            filename = hashlib.sha1(f.read()).hexdigest()[:16]

    response = {
        "status": 0,
        "image_url": f"{address}/static/{device['image']}",
        "filename": filename,
        "update_firmware": update_firmware,
        "firmware_url": firmware_url if update_firmware else None,
        "refresh_rate": str(refresh_rate),
        "reset_firmware": False,
    }

    logger.debug(f"/api/display response: {response}")
    return response


# Accept both /api/logs and /api/log
@app.post("/api/logs", include_in_schema=True)
@app.post("/api/log", include_in_schema=False)
async def receive_logs(request: Request):
    data = await request.json()
    logs.append(data)
    logger.info(f"Log received: {data}")
    return {"status": 200, "message": "Log received"}


@app.get("/static/{filename}")
async def serve_image(filename: str):
    file_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.warning(f"Image not found: {file_path}")
        return JSONResponse(status_code=404, content={"status": 404, "error": "Image not found"})
    logger.info(f"Serving image: {file_path}")
    return FileResponse(file_path)

@app.get("/firmware/{filename}")
async def serve_firmware(filename: str):
    file_path = os.path.join(FIRMWARE_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.warning(f"Firmware not found: {file_path}")
        return JSONResponse(status_code=404, content={"status": 404, "error": "Firmware not found"})
    logger.info(f"Serving firmware: {file_path}")
    return FileResponse(file_path, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Device API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=2300, help="Port to bind")
    parser.add_argument("--loglevel", type=str, default="debug", help="Logging level (debug, info, warning, error, critical)")
    parser.add_argument("--address", type=str, default="http://192.168.0.100:2300", help="Remote serving address for clients")
    parser.add_argument("--logfile", type=str, default=os.path.join(DATA_FOLDER, "trmnl.log"), help="Log file, written in addition to the terminal")
    args = parser.parse_args()

    # Configure logging with argparse loglevel
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")
    # Log to the terminal and to data/trmnl.log (untracked, so it survives restarts and updates).
    # 5 files of 2 MB are kept.
    log_handlers = [logging.StreamHandler()]
    try:
        log_handlers.append(RotatingFileHandler(args.logfile, maxBytes=2_000_000, backupCount=4))
    except OSError as e:
        print(f"Not logging to {args.logfile}: {e}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s", handlers=log_handlers)
    logger.setLevel(numeric_level)

    # Load devices from disk at startup
    load_devices()

    logger.info(f"Starting server on {args.host}:{args.port} with loglevel={args.loglevel}")
    address = args.address
    # log_config=None: uvicorn's own messages and access log go through the handlers above too
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.loglevel, log_config=None)

