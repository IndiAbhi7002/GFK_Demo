# === SCRIPT: main S3 multiprocessing fetcher (folder-based, per-image parallel) ===
import boto3
import logging
from io import BytesIO
import time
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count, Manager, set_start_method
import threading
import cv2
import signal
import sys
import numpy as np
import atexit
import pytz
import torch

from processing_files.configuration import (
    S3_BUCKET, S3_PREFIX, S3_LOG_KEY,
    s3_access_key, s3_secret_key, s3_region_name,
    DATA_BASE, logger, log_file,
    old_model_content_path, platform_logo_model_path, new_model_blg_demo_path,youtube_model_path,phash_dataset_path
)

from processing_files.test_model_linux3 import process_image_pipeline
from ultralytics import YOLO

# === Set fixed start date for filtering (Nepal time) ===
def get_start_date_nepal():
    nepal = pytz.timezone("Asia/Kathmandu")
    return datetime(2025, 7, 20, 0, 0, 0, tzinfo=nepal)

# === Global vars ===
shutdown_event = threading.Event()
pool = None
log_thread = None

# === Single target folder to fetch images from ===
S3_IMAGE_FOLDER = S3_PREFIX # Example: 'incoming-images/' or 'images/'

# === Shared YOLO models ===
platform_logo_model = None
old_model_content_model = None
new_model_blg_demo = None
youtube_model = None
pash_model = None

# === Initialize YOLO models inside workers ===
def init_worker():
    global old_model_content_model, platform_logo_model, new_model_blg_demo,youtube_model,phash_model
    platform_logo_model = YOLO(platform_logo_model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    old_model_content_model = YOLO(old_model_content_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    new_model_blg_demo = YOLO(new_model_blg_demo_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    youtube_model = YOLO(youtube_model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
    # phash_model = np.load(phash_dataset_path, allow_pickle=True).item()

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except ValueError:
        pass

# === S3 client ===
s3 = boto3.client('s3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    region_name=s3_region_name
)

# === Signal + Cleanup Handling ===
def signal_handler(signum, frame):
    print(f"\n[INFO] Received signal {signum}. Initiating graceful shutdown...")
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()
    cleanup_and_exit()

def cleanup_and_exit():
    global pool, log_thread
    if shutdown_event.is_set():
        return

    print("[INFO] Cleaning up resources...")
    logger.info("Cleaning up resources...")
    shutdown_event.set()

    if pool:
        print("[INFO] Terminating worker pool...")
        logger.info("Terminating worker pool...")
        try:
            pool.terminate()
            pool.join(timeout=5)
        except Exception as e:
            print(f"[WARNING] Pool termination error: {e}")
        print("[INFO] Worker pool terminated.")

    if log_thread and log_thread.is_alive():
        log_thread.join(timeout=3)

    try:
        s3.upload_file(log_file, S3_BUCKET, S3_LOG_KEY)
        print("[INFO] Final log upload completed.")
    except Exception as e:
        print(f"[ERROR] Final log upload failed: {e}")

    print("[INFO] Shutdown complete. Exiting.")
    logger.info("Shutdown complete. Exiting.")
    os._exit(0)

def upload_logs_to_s3():
    while not shutdown_event.is_set():
        try:
            if not shutdown_event.wait(60):
                if not shutdown_event.is_set():
                    s3.upload_file(log_file, S3_BUCKET, S3_LOG_KEY)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.error(f"Failed to upload log file to S3: {e}")
    print("[INFO] Log upload thread terminated.")
    logger.info("Log upload thread terminated.")

# === Per-image processing function (executed by each worker) ===
def process_image_task(key):
    try:
        filename = os.path.basename(key)
        if not filename.endswith(".jpg"):
            return

        parts = filename.split(".")[0].split("_")
        if len(parts) != 2:
            logger.warning(f"Unexpected filename format: {filename}")
            return

        device_id_str, timestamp_str = parts
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            logger.warning(f"Invalid timestamp in filename: {filename}")
            return

        obj = s3.head_object(Bucket=S3_BUCKET, Key=key)
        last_modified_utc = obj.get("LastModified")
        if not last_modified_utc:
            logger.warning(f"No LastModified for {key}")
            return

        last_modified_nepal = last_modified_utc.astimezone(pytz.timezone("Asia/Kathmandu"))
        if last_modified_nepal < get_start_date_nepal():
            return

        logger.info(f"Processing file: {key} (LastModified: {last_modified_nepal})")
        start_fetch = time.time()
        img_response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        img_data = img_response["Body"].read()
        np_img = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"Failed to decode image for {key}")
            return

        metadata = {
            "device_id": device_id_str,
            "timestamp": timestamp,
        }

        fetch_time_s3 = round(time.time() - start_fetch, 3)
        logger.info(f"Fetched  image {key} in {fetch_time_s3} seconds")
        process_image_pipeline(image, metadata, DATA_BASE)
        fetch_time = round(time.time() - start_fetch, 3)
        logger.info(f"Fetched and processed image {key} in {fetch_time} seconds")

        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=key)
            logger.info(f"Processed and deleted image: {key}")
        except Exception as e:
            logger.error(f"Failed to delete image: {key} - {str(e)}")

    except Exception as e:
        logger.error(f"Error processing image {key}: {e}")

# === Main entry point ===
if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_and_exit)

    manager = Manager()

    log_thread = threading.Thread(target=upload_logs_to_s3, daemon=False)
    log_thread.start()

    try:
        pool = Pool(processes=min(8, cpu_count()), initializer=init_worker)
    except Exception as e:
        logger.error(f"Failed to create process pool: {e}")
        sys.exit(1)

    logger.info("Starting main processing loop...")
    print("[INFO] Starting main processing loop... Press Ctrl+C to stop")

    while not shutdown_event.is_set():
        try:
            files = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_IMAGE_FOLDER).get("Contents", [])
            image_keys = [obj["Key"] for obj in files if obj["Key"].endswith(".jpg")]

            if not image_keys:
                logger.info("No images found.")
            else:
                logger.info(f"Found {len(image_keys)} images. Distributing for processing.")
                pool.map_async(process_image_task, image_keys).get(timeout=60)

            if shutdown_event.wait(10):
                break

        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt detected")
            break
        except Exception as e:
            logger.error(f"Unexpected main loop error: {e}")
            if shutdown_event.wait(5):
                break

    cleanup_and_exit()
