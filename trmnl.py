import os
import uuid
import hashlib
import logging
import argparse
import json
import subprocess
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
    "firmware_upgrading": False,
    "firmware_version": "1.5.2",
    "firmware_download_url": "https://trmnl.s3.us-east-2.amazonaws.com/path-to-firmware.bin",
    "default_image": "default.bmp",
    "default_refresh_rate": "1800",
    "pre_refresh": 20,
}

# Ensure images folder exists
IMAGE_FOLDER = "images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)


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

async def run_update_script_later(device, delay):
    await asyncio.sleep(delay)
    script_path = os.path.join("scripts", f"{device['friendly_id']}.sh")
    if os.path.exists(script_path):
        try:
            logger.info(f"Running update script for {device['friendly_id']} after {delay}s")
            subprocess.run(["python3", script_path], check=True)
            new_image = f"{device['friendly_id']}.bmp"
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
):
    logger.debug(f"/api/display called with headers: ID={ID}, Access_Token={Access_Token}, Refresh_Rate={Refresh_Rate}, Battery_Voltage={Battery_Voltage}, FW_Version={FW_Version}, RSSI={RSSI}")

    if not ID or not Access_Token:
        logger.warning("/api/display missing required headers")
        return JSONResponse(status_code=400, content={"status": 400, "error": "Missing headers"})

    device = devices.get(ID)
    if not device or device["api_key"] != Access_Token:
        logger.error(f"Device not found or token mismatch for ID={ID}")
        return {"status": 500, "error": "Device not found"}
        
    # Update the image dynamically if a script exists
    # Schedule update script execution (refresh_rate - 20s)
    try:
        refresh_delay = int(device.get("refresh_rate", SETTINGS["default_refresh_rate"]))
        refresh_delay -= int(device.get("pre_refresh", SETTINGS["pre_refresh"]))
        if refresh_delay > 0:
            asyncio.create_task(run_update_script_later(device, refresh_delay))
    except Exception as e:
        logger.error(f"Failed to schedule script execution for {device['friendly_id']}: {e}")

    # Determine firmware update requirement
    update_firmware = SETTINGS["firmware_upgrading"] and FW_Version != SETTINGS["firmware_version"]

    file_path = os.path.join(IMAGE_FOLDER, device["image"])
    if os.path.exists(file_path):
        file_timestamp = os.path.getmtime(file_path)
        filename = str(int(file_timestamp))  # or use datetime if you prefer ISO format

    response = {
        "status": 0,
        "image_url": f"{address}/static/{device['image']}",
        "filename": filename,
        "update_firmware": update_firmware,
        "firmware_url": SETTINGS["firmware_download_url"] if update_firmware else None,
        "refresh_rate": device.get("refresh_rate", SETTINGS["default_refresh_rate"]),
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

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Device API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=2300, help="Port to bind")
    parser.add_argument("--loglevel", type=str, default="debug", help="Logging level (debug, info, warning, error, critical)")
    parser.add_argument("--address", type=str, default="http://192.168.0.100:2300", help="Remote serving address for clients")
    args = parser.parse_args()

    # Configure logging with argparse loglevel
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(numeric_level)

    # Load devices from disk at startup
    load_devices()

    logger.info(f"Starting server on {args.host}:{args.port} with loglevel={args.loglevel}")
    address = args.address
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.loglevel)

