# === Image processing script===
import os
import cv2
import numpy as np
import re
import time
import torch
from ultralytics import YOLO
import boto3
import logging
from typing import Tuple, List, Optional
import random
from processing_files.configuration import (
    S3_BUCKET, s3_access_key, s3_secret_key, s3_region_name,
    S3_RECOGNIZED_PREFIX, S3_UNRECOGNIZED_PREFIX,
    old_model_content_path, platform_logo_model_path, new_model_blg_demo_path,youtube_model_path,logger,phash_dataset_path,
    YT_MODEL_LIST,PLATFORM_LIST, CONTENT_LIST, ADS_LIST,OLD_MODEL_EXCLUDE_LIST,OLD_MODEL_CONTENT_LIST,OLD_MODEL_TV_OTT_YT_LIST,OUTPUT_IMAGE_QUALITY
)
from processing_files.Database_connection import (
    insert_detection_to_db, batch_insert_detections_to_db,
    mqtt_publish_recognized_detection, mqtt_publish_unreognized_detection,
    local_mongo_db_connection_recognized_detection, local_mongo_db_connection_unrecognized_detection
)

import easyocr
# from paddleocr import PaddleOCR
# Initialize the OCR reader once

TV_CHANNEL_CONFIDENCE_THRESHOLD = 0.60
TV_CHANNEL_IOU_THRESHOLD = 0.45

ADVERTISEMENT_CONFIDENCE_THRESHOLD = 0.80
ADVERTISEMENT_IOU_THRESHOLD = 0.45

OLD_CONTENT_CONFIDENCE_THRESHOLD = 0.60
OLD_CONTENT_IOU_THRESHOLD = 0.45

NEW_CONTENT_CONFIDENCE_THRESHOLD = 0.80
NEW_CONTENT_IOU_THRESHOLD = 0.45

YOUTUBE_CONFIDENCE_THRESHOLD = 0.60
YOUTUBE_IOU_THRESHOLD = 0.45


# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Initialize global models (these will be set in the main script)
platform_logo_model = None
old_model_content_model = None
new_model_blg_demo = None  # Placeholder for the unified model
youtube_model = None
phash_model = None

s3_client = boto3.client(
    's3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    region_name=s3_region_name
)

def get_unique_color(class_id):
    random.seed(class_id)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
# EXCLUDED_ADVERTISEMENT_LABELS = ["Himalaya TV", "KantipurMax"]
os.makedirs(S3_RECOGNIZED_PREFIX, exist_ok=True)
os.makedirs(S3_UNRECOGNIZED_PREFIX, exist_ok=True)

# Load OCR once (move to top-level or global setup)
# paddle_ocr_reader = PaddleOCR(device="gpu" ,ocr_version="PP-OCRv4",use_angle_cls=False, lang='en')

def upload_to_s3(file_path: str, s3_key: str) -> bool:
    try:
        s3_client.upload_file(Filename=file_path, Bucket=S3_BUCKET, Key=s3_key)
        logger.info(f"Uploaded to S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}: {s3_key}")
        return False
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting local file {file_path}: {e}")


def extract_text_from_image_easyocr(image):
    """
    Extracts text from a given image using EasyOCR and returns the text and average confidence.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: (extracted_text (str), average_confidence (float))
    """
    reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if you don't have a GPU

    start_time=time.time()
    try:
        ocr_detections = []
        results = reader.readtext(image, detail=1)
        if not results:
            return "", 0.0

        extracted_texts = [text for (_, text, conf) in results]
        confidences = [conf for (_, text, conf) in results]

        avg_confidence = sum(confidences) / len(confidences)
        full_text = ' '.join(extracted_texts).strip()
        avg_confidence=float(round(avg_confidence, 2))
        if avg_confidence >=0.5:
            print(f"avg_confidence : {avg_confidence}")
            ocr_detections.append((full_text,avg_confidence))
            logger.info(f"Ocr detection time: {time.time() - start_time:.4f}s, Found Ocr : {full_text}")
            return image,ocr_detections

    except Exception as e:
        return f"Error reading image: {e}", 0.0


# def extract_text_from_image_paddleocr(image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
#     """
#     Uses PaddleOCR to extract text from full image.
#     Returns the image and OCR results (text + confidence).
#     """
#     start_time = time.time()

#     try:
#         # Convert to BGR if needed (Paddle expects BGR or path)
#         if image.shape[-1] == 4:  # remove alpha if exists
#             image = image[:, :, :3]

#         result = paddle_ocr_reader.ocr(image, cls=False)

#         ocr_detections = []
#         if result and isinstance(result[0], list):
#             for line in result[0]:
#                 text = line[1][0]
#                 conf = float(round(line[1][1], 2))
#                 if conf >= 0.5:
#                     ocr_detections.append((text, conf))

#         logger.info(f"PaddleOCR detection time: {time.time() - start_time:.4f}s, Found OCR: {ocr_detections}")
#         return image, ocr_detections

#     except Exception as e:
#         logger.error(f"PaddleOCR failed: {e}")
#         return image, []


def youtube_detection(frame: np.ndarray) -> List[dict]:
    global youtube_model
    if youtube_model is None:
        youtube_model = YOLO(youtube_model_path).to(device)
    start_time = time.time()

    try:
        results = youtube_model(frame, device=device, conf=YOUTUBE_CONFIDENCE_THRESHOLD, iou=YOUTUBE_IOU_THRESHOLD, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]
                # print(f"label : {label}")
                class_id = int(box.cls[0])
                if confidence > YOUTUBE_CONFIDENCE_THRESHOLD:
                    detections.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id
                    })
        logger.info(f"Youtube detection time: {time.time() - start_time:.4f}s")
        return detections
    except Exception as e:
        logger.error(f"Error in YouTube detection: {e}")
        return []

def detect_platform(frame: np.ndarray) -> List[dict]:
    global platform_logo_model
    if platform_logo_model is None:
        platform_logo_model = YOLO(platform_logo_model_path).to(device)
    start_time = time.time()

    try:
        results = platform_logo_model(frame, device=device, conf=TV_CHANNEL_CONFIDENCE_THRESHOLD, iou=TV_CHANNEL_IOU_THRESHOLD, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]
                class_id = int(box.cls[0])
                if confidence > TV_CHANNEL_CONFIDENCE_THRESHOLD:
                    detections.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id
                    })
        logger.info(f"Only Paltform TV channel detection time: {time.time() - start_time:.4f}s")
        return detections
    except Exception as e:
        logger.error(f"Error in platform detection: {e}")
        return []

def detect_old_content(frame: np.ndarray) -> Tuple[List[dict], List[dict]]:
    global old_model_content_model
    if old_model_content_model is None:
        old_model_content_model = YOLO(old_model_content_path).to(device)

    old_tv_ott = []
    old_content = []
    start_time = time.time()

    try:
        results = old_model_content_model(frame, device=device, conf=OLD_CONTENT_CONFIDENCE_THRESHOLD, iou=OLD_CONTENT_IOU_THRESHOLD, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]
                class_id = int(box.cls[0])
                if label in OLD_MODEL_EXCLUDE_LIST:
                    continue
                detection = {
                    'label': label,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id
                }
                if label in OLD_MODEL_TV_OTT_YT_LIST:
                    old_tv_ott.append(detection)
                elif label in OLD_MODEL_CONTENT_LIST:
                    old_content.append(detection)
        logger.info(f"Old content detection time: {time.time() - start_time:.4f}s")
        return old_content, old_tv_ott
    except Exception as e:
        logger.error(f"Error in old content detection: {e}")
        return [], []

def detect_new_content(frame: np.ndarray) -> Tuple[List[dict], List[dict], List[dict]]:
    global new_model_blg_demo
    if new_model_blg_demo is None:
        new_model_blg_demo = YOLO(new_model_blg_demo_path).to(device)

    new_ads = []
    new_platform = []
    new_content = []
    start_time = time.time()

    try:
        results = new_model_blg_demo(frame, device=device, conf=NEW_CONTENT_CONFIDENCE_THRESHOLD, iou=NEW_CONTENT_IOU_THRESHOLD, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]
                class_id = int(box.cls[0])

                detection = {
                    'label': label,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id
                }

                if label in ADS_LIST:
                    new_ads.append(detection)
                elif label in PLATFORM_LIST:
                    new_platform.append(detection)
                elif label in CONTENT_LIST:
                    new_content.append(detection)
        logger.info(f" New Model Detection time: {time.time() - start_time:.4f}s")
        return new_platform, new_ads, new_content
    except Exception as e:
        logger.error(f"Error in new content detection: {e}")
        return [], [], []

HASH_SIZE = 8
DISTANCE_THRESHOLD = 10
import imagehash
from PIL import Image
from datetime import datetime, timedelta

def find_best_pash_match(frame_hash, phash_dataset):
    best_title, best_time, best_distance = None, None, 9999
    for content_id, content_data in phash_dataset.items():
        for h, ts in content_data["hashes"]:
            dist = frame_hash - h
            if dist < best_distance:
                best_distance, best_title, best_time = dist, content_id, ts
    return best_title, best_time, best_distance

# Replace your current phash_detection with this:

def phash_detection(frame: np.ndarray) -> List[Tuple[str, float]]:
    """
    Returns a list of (label, confidence) tuples.
    label: "best_title-<HH:MM:SS>" (from timedelta)
    confidence: mapped from distance to [0..1], higher is better
    """
    global phash_model
    if phash_model is None:
        phash_model = np.load(phash_dataset_path, allow_pickle=True).item()
        logger.info("Loaded phash_model in worker")

    try:
        frame_hash = imagehash.phash(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            hash_size=HASH_SIZE
        )
        best_title, best_time, best_distance = find_best_pash_match(frame_hash, phash_model)
        logger.info(f"pHash detection time: Found pHash : {best_title} with distance {best_distance}")

        if not best_title:
            return []

        if best_distance <= DISTANCE_THRESHOLD:
            position = str(timedelta(seconds=int(best_time)))  # HH:MM:SS
            label = f"{best_title}-{position}"
            # map distance -> confidence in [0..1]; closer (smaller) distance = higher confidence
            conf = max(0.0, 1.0 - (best_distance / (DISTANCE_THRESHOLD + 1.0)))
            conf = round(conf, 2)
            return [(label, conf)]
        else:
            return []
    except Exception as e:
        logger.error(f"Error in pHash detection: {e}")
        return []


def save_processed_image_and_store_db(frame: np.ndarray, metadata: dict, detections: dict, original_size: Tuple[int, int], DATA_BASE: str) -> dict:
    timestamp_str = str(metadata['timestamp'])
    sanitized_timestamp = re.sub(r'[-:.]', '_', timestamp_str)
    device_id = str(metadata['device_id'])  # Move this early so it's always available

    # original_width, original_height = original_size
    resized_frame = cv2.resize(frame, (640, 640))
    ocr_detections = detections.get('Ocr',[])
    channel_detections = detections.get('channels', [])
    ad_detections = detections.get('advertisements', [])
    new_platform_detections = detections.get('new_platform', [])
    new_content_detections = detections.get('new_content', [])
    old_content_detections = detections.get('old_content', [])
    old_platform_detections = detections.get('old_platform', [])
    youtube_detections = detections.get('youtube', [])
    # phash_detections = detections.get('phash', [])
    # has_detections = bool(ocr_detections or channel_detections or ad_detections or new_platform_detections or new_content_detections or old_content_detections or old_platform_detections)

    has_detections = bool(channel_detections  or youtube_detections or ad_detections or new_platform_detections or new_content_detections or old_content_detections or old_platform_detections)#phash_detections

    # has_detections = bool(channel_detections or ad_detections)

    if has_detections:
        filename = f"{device_id}_{sanitized_timestamp}_recognized.jpg"
        device_folder = os.path.join(S3_RECOGNIZED_PREFIX, device_id)
        s3_folder = f"{S3_RECOGNIZED_PREFIX}{device_id}/"
        category = "recognized"
    else:
        filename = f"{device_id}_{sanitized_timestamp}_unrecognized.jpg"
        device_folder = os.path.join(S3_UNRECOGNIZED_PREFIX, device_id)
        s3_folder = f"{S3_UNRECOGNIZED_PREFIX}{device_id}/"
        category = "unrecognized"

    os.makedirs(device_folder, exist_ok=True)
    local_path = os.path.join(device_folder, filename)
    s3_key = f"{s3_folder}{filename}"

    result = {'image_saved': False, 'db_insertions': 0, 's3_key': None, 'category': category}

    try:
        if cv2.imwrite(local_path, resized_frame,[cv2.IMWRITE_JPEG_QUALITY, OUTPUT_IMAGE_QUALITY]):
            upload_success = upload_to_s3(local_path, s3_key)
            # upload_success = False
            if upload_success:
                result['image_saved'] = True
                result['s3_key'] = s3_key
            else:
                return result

            if has_detections:
                record = {
                    'ts': metadata['timestamp'],
                    'device_id': device_id,
                    'ocr': ocr_detections,
                    'channels': channel_detections,
                    'ads': ad_detections,
                    'new_platform': new_platform_detections,
                    'new_content': new_content_detections,
                    'old_content': old_content_detections,
                    'old_platform': old_platform_detections,
                    'youtube': youtube_detections,
               
                    'image_path': s3_key
                }     # 'phash': phash_detections,

                if DATA_BASE == "postgres":
                    inserted_count = batch_insert_detections_to_db(record)
                elif DATA_BASE == "MQTT":
                    inserted_count = mqtt_publish_recognized_detection(record)
                elif DATA_BASE == "Local MongoDB":
                    inserted_count = local_mongo_db_connection_recognized_detection(record)

                result['db_insertions'] = inserted_count
            else:
                record = {
                    'ts': metadata['timestamp'],
                    'device_id': device_id,
                    'Ocr': ocr_detections,
                    'channels': [],
                    'ads': [],
                    'new_platform': [],
                    'new_content': [],
                    'old_content': [],
                    'old_platform': [],
                    'youtube': [],
                    'phash': [],
                    'image_path': s3_key
                }

                if DATA_BASE == "postgres":
                    success = batch_insert_detections_to_db(record)
                elif DATA_BASE == "MQTT":
                    success = mqtt_publish_unreognized_detection(record)
                elif DATA_BASE == "Local MongoDB":
                    success = local_mongo_db_connection_unrecognized_detection(record)

                if success:
                    result['db_insertions'] = 1
        else:
            logger.error(f"Failed to save image: {local_path}")
    except Exception as e:
        logger.error(f"Error in save_processed_image_and_store_db for device {device_id}: {e}")
    return result

def process_image_pipeline(frame: np.ndarray, metadata: dict, DATA_BASE: str) -> dict:
    pipeline_start = time.time()
    original_height, original_width = frame.shape[:2]
    original_size = (original_width, original_height)
    try:
        # Clean copies for model input
        # paddle__ocr_detections = extract_text_from_image_paddleocr(frame.copy())
        # print(f"paddle__ocr_detections: {paddle__ocr_detections}")
        platform_detections = detect_platform(frame.copy())
        old_content_detections, old_platform_detections = detect_old_content(frame.copy())
        new_platform_detections, new_ads_detections, new_content_detections = detect_new_content(frame.copy())
        youtube_detections = youtube_detection(frame.copy())
        # phash_detections = phash_detection(frame.copy())
        ocr_detections = [("", 0.0)]

        # Create a unified annotated frame
        processed_frame = frame.copy()

        def draw_detections(frame, detections):
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = det['label']
                confidence = det['confidence']
                class_id = det['class_id']
                color = get_unique_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw all in one step
        draw_detections(processed_frame, platform_detections)
        draw_detections(processed_frame, old_platform_detections)
        draw_detections(processed_frame, old_content_detections)
        draw_detections(processed_frame, new_platform_detections)
        draw_detections(processed_frame, new_ads_detections)
        draw_detections(processed_frame, new_content_detections)
        draw_detections(processed_frame, youtube_detections)

        # Store detection lists for DB/log
        detections = {
            'Ocr': ocr_detections,
            'channels': [(d['label'], d['confidence']) for d in platform_detections],
            'advertisements': [(d['label'], d['confidence']) for d in new_ads_detections],
            'new_platform': [(d['label'], d['confidence']) for d in new_platform_detections],
            'new_content': [(d['label'], d['confidence']) for d in new_content_detections],
            'old_content': [(d['label'], d['confidence']) for d in old_content_detections],
            'old_platform': [(d['label'], d['confidence']) for d in old_platform_detections],
            'youtube': [(d['label'], d['confidence']) for d in youtube_detections],
            }#'phash': phash_detections
        # print(f"detections : {detections}")
        save_result = save_processed_image_and_store_db(processed_frame, metadata, detections, original_size, DATA_BASE)

        return {
            'device_id': metadata['device_id'],
            'timestamp': metadata['timestamp'],
            'channel_detections': detections['channels'],
            'advertisement_detections': detections['advertisements'],
            'new_platform_detections': detections['new_platform'],
            'new_content_detections': detections['new_content'],
            'old_content_detections': detections['old_content'],
            'old_platform_detections': detections['old_platform'],
            'youtube_detections': detections['youtube'],
            's3_key': save_result.get('s3_key'),
            'image_saved': save_result.get('image_saved', False),
            'db_insertions': save_result.get('db_insertions', 0),
            'category': save_result.get('category'),
            'processing_time': time.time() - pipeline_start,
            'success': True
        }

    except Exception as e:
        logger.error(f"Pipeline error for device {metadata['device_id']}: {e}")
        return {
            'device_id': metadata['device_id'],
            'timestamp': metadata['timestamp'],
            'error': str(e),
            'success': False,
            'processing_time': time.time() - pipeline_start
        }

def tv_channels_image_data(frame: np.ndarray, metadata: dict) -> dict:
    return process_image_pipeline(frame, metadata)

def process_multiple_images(image_data_list: List[Tuple[np.ndarray, dict]]) -> List[dict]:
    return [process_image_pipeline(frame, metadata) for frame, metadata in image_data_list]
