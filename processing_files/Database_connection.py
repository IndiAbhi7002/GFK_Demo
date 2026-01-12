## Databse insertion script
from processing_files.configuration import (
    # Add these to your configuration file
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT,logger,DB_TYPE, S3_BASE_URL
    ,MQTT_PORT, MQTT_TOPIC, CA_CERT_PATH, CLIENT_CERT_PATH,CLIENT_KEY_PATH, MQTT_BROKER,MONGODB_CONNECTION_STRING
)
from typing import Tuple, List, Optional
import paho.mqtt.client as mqtt
import pymongo
import json
import psycopg2
import logging
import json

# Create an MQTT client instance
mqtt_client = mqtt.Client()

# Setup TLS/SSL connection with certificates
mqtt_client.tls_set(ca_certs=CA_CERT_PATH, certfile=CLIENT_CERT_PATH, keyfile=CLIENT_KEY_PATH)

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code: {rc}")

# mqtt_client.loop_start()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected to AWS IoT")
    else:
        logging.error(f"Connection failed with code {rc}")

def on_publish(client, userdata, mid):
    logging.info(f"Message {mid} published.")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        logging.warning(f"Unexpected disconnection. Result code: {rc}")

# Set the callback for connection
mqtt_client.on_connect = on_connect
mqtt_client.on_connect = on_connect
mqtt_client.on_publish = on_publish
mqtt_client.on_disconnect = on_disconnect

# Connect to the broker
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# PostgresSQL Database connection 
def get_db_connection():
    """Get database connection with error handling."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None
    
# MongoDB connection
def get_mongo_connection() -> Optional[pymongo.MongoClient]:
    """
    Get MongoDB connection with error handling.

    Returns:
        pymongo.MongoClient instance or None if connection fails.
    """
    try:
        client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
        return client
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}")
        return None

#== MQTT and Database Functions ===
# Publish detection result to MQTT broker for unrecognized detections
def mqtt_publish_unreognized_detection(ts: int, device_id: int, channel_name: str, score: float, image_path: str) -> bool:
    """
    Publish detection result to MQTT broker.

    Args:
        ts: Timestamp
        device_id: Device ID
        channel_name: Detected channel or advertisement name
        score: Confidence score (rounded to 2 decimal places)
        image_path: S3 image path with base URL

    Returns:
        True if successful, False otherwise
    """
    try:
        # Construct full S3 URL
        full_image_path = f"{S3_BASE_URL}{image_path}"

        # Round score to 2 decimal places
        rounded_score = round(score, 2)

        # if channel_name == "Unrecognized":
        #     DB_TYPE = 33

        # Create the message payload
        payload = {
        "TS": ts,
        "Type": 33,
        "DEVICE_ID": device_id,
        "Details": {
            "Channel_name": channel_name,
            "score": rounded_score,
            "image_path":full_image_path
                    }
                }

        # Publish the message to the MQTT topic
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        logger.info(f"Published to MQTT: {channel_name} (score: {rounded_score}) for device {device_id}")
        return True

    except Exception as e:
        logger.error(f"MQTT publish error for {channel_name}: {e}")
        return False

# Batch publish detection results to MQTT broker
def mqtt_publish_recognized_detection(detections_data: List[dict]) -> int:
    """
    Publish detection result to MQTT broker.
    Args:
        ts: Timestamp
        device_id: Device ID
        channel_name: Detected channel or advertisement name
        score: Confidence score (rounded to 2 decimal places)
        image_path: S3 image path with base URL
        Returns:
        True if successful, False otherwise
        """
    if not detections_data:
        return 0

    inserted_count = 0

    try:
        for detection in detections_data:
            try:
                # Construct full S3 URL
                full_image_path = f"{S3_BASE_URL}{detection['image_path']}"

                # Round score to 2 decimal places
                rounded_score = round(detection['score'], 2)

                # Create the message payload
                payload = {
                "TS": detection['ts'],
                "Type": DB_TYPE,
                "DEVICE_ID": detection['device_id'],
                "Details": {
                    "Channel_name": detection['channel_name'],
                    "score": rounded_score,
                    "image_path": full_image_path
                            }
                        }

                inserted_count += 1

                # Publish the message to the MQTT topic
                mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
                logger.info(f"Published to MQTT: {detection['channel_name']} (score: {rounded_score}) for device {detection['device_id']}")
                return True

            except Exception as e:
                logger.error(f"Error inserting detection {detection.get('channel_name', 'unknown')}: {e}")
                continue

        logger.info(f"Batch isended {inserted_count} detections to MQTT broker")

    except Exception as e:
        logger.error(f"Batch mqtt sending error: {e}")

#MongoDB connection for unrecognized detection
def local_mongo_db_connection_unrecognized_detection(ts: int, device_id: int, channel_name: str, score: float, image_path: str) -> bool:
    client = get_mongo_connection()
    if not client:
        return False
    try:
        # Access the database and collection
        db = client['Nepal']
        collection = db['indi_test']
        # Construct full S3 URL
        full_image_path = f"{S3_BASE_URL}{image_path}"

        # Round score to 2 decimal places
        rounded_score = round(score, 2)

        # if channel_name == "Unrecognized":
        #     DB_TYPE = 33

        # Create the message payload
        payload = {
        "TS": ts,
        "Type": 33,
        "DEVICE_ID": device_id,
        "Details": {
            "Channel_name": channel_name,
            "score": rounded_score,
            "image_path":full_image_path
                    }
                }

        # Insert the document into the collection
        collection.insert_one(payload)
        logger.info(f"Inserted to Local Mongo DB: {channel_name} (score: {rounded_score}) for device {device_id}")
        return True

    except Exception as e:
        logger.error(f"Local Mongo DB Database insert error for {channel_name}: {e}")
        return False

#MongoDB connection for recognized detection
def local_mongo_db_connection_recognized_detection(detections_data: List[dict]):
    """
    Insert multiple detection results in a single transaction.

    Args:
        detections_data: List of detection dictionaries

    Returns:
        Number of successfully inserted records
    """
    if not detections_data:
        return 0

    client = get_mongo_connection()
    if not client:
        return False


    inserted_count = 0

    try:
        for detection in detections_data:
            try:
                # Access the database and collection
                db = client['Nepal']
                collection = db['indi_test']

                # Construct full S3 URL
                full_image_path = f"{S3_BASE_URL}{detection['image_path']}"

                # Round score to 2 decimal places
                rounded_score = round(detection['score'], 2)

                PAYLOAD = {
                    "TS": detection['ts'],
                    "Type": DB_TYPE,
                    "DEVICE_ID": detection['device_id'],
                    "Details": {
                        "Channel_name": detection['channel_name'],
                        "score": rounded_score,
                        "image_path": full_image_path
                    }
                }
                collection.insert_one(PAYLOAD)
                logger.info(f"Inserted to Local Mongo DB: {detection['channel_name']} (score: {rounded_score}) for device {detection['device_id']}")

                inserted_count += 1

            except Exception as e:
                logger.error(f"Local Mongo DB Error inserting detection {detection.get('channel_name', 'unknown')}: {e}")
                continue

        logger.info(f" Batch inserted {inserted_count} detections to Local Mongo DB ")

    except Exception as e:
        logger.error(f"Batch Local Mongo DB  insert error: {e}")

    return inserted_count



#Post greSQL Database connection
def insert_detection_to_db(ts: int, device_id: str, channel_name: str, score: float, image_path: str) -> bool:
    """
    Insert detection result into PostgreSQL database.

    Args:
        ts: Timestamp
        device_id: Device ID
        channel_name: Detected channel or advertisement name
        score: Confidence score (rounded to 2 decimal places)
        image_path: S3 image path with base URL

    Returns:
        True if successful, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            # Construct full S3 URL
            full_image_path = f"{S3_BASE_URL}{image_path}"

            # Round score to 2 decimal places
            rounded_score = round(score, 2)
            if channel_name == "Unrecognized":
                DB_TYPE=33

            # Prepare JSON details
            details = json.dumps({
                "channel_name": channel_name,
                "score": rounded_score,
                "image_path": full_image_path
            })

            cur.execute("""
                INSERT INTO events (timestamp, type, deviceid,details)
                VALUES (%s, %s, %s, %s)
            """, (
                ts,
                DB_TYPE,
                device_id,
                details

            ))

            conn.commit()
            logger.info(f"Inserted to DB: {channel_name} (score: {rounded_score}) for device {device_id}")
            return True

    except Exception as e:
        logger.error(f"Database insert error for {channel_name}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()



import time  # required to track duration

def batch_insert_detections_to_db(detection: dict) -> int:
    """
    Insert a normalized detection record:
    - 1 row in `Event`
    - N rows in `EventChannel`
    - M rows in `EventAd`
    - P rows in `EventContent`

    Args:
        detection: dict with keys:
          ['device_id', 'ts', 'image_path', 'channels', 'ads', 'new_platform', 'new_content', 'old_content']

    Returns:
        1 if Event inserted successfully, else 0
    """
    conn = get_db_connection()
    if not conn:
        return 0

    inserted = 0
    start_time = time.time()

    try:
        with conn.cursor() as cur:
            device_id = detection['device_id']
            timestamp = detection['ts']
            full_image_path = f"{S3_BASE_URL}{detection['image_path']}"

            # # Detection groups
            # channel_detections = detection.get("channels", [])
            # ad_detections = detection.get("ads", [])
            # new_platform_detections = detection.get("new_platform", [])
            # new_content_detections = detection.get("new_content", [])
            # old_content_detections = detection.get("old_content", [])

            # Ensure all detection lists are proper lists
            ocr_detections          = list(detection.get("ocr",[]))
            channel_detections      = list(detection.get("channels", []))
            ad_detections           = list(detection.get("ads", []))
            new_platform_detections = list(detection.get("new_platform", []))
            new_content_detections  = list(detection.get("new_content", []))
            old_content_detections  = list(detection.get("old_content", []))
            old_platform_detections = list(detection.get("old_platform", []))
            youtube_detections      = list(detection.get("youtube", []))
            # phash_detections        = list(detection.get("phash", []))

            # Determine DB type
            all_scores = [score for _, score in (
                channel_detections + ad_detections +
                new_platform_detections + new_content_detections +
                old_content_detections + old_platform_detections+ocr_detections+youtube_detections
            )]#+ phash_detections

            max_score = round(max(all_scores), 2) if all_scores else 0.0
            DB_TYPE = 29 if all_scores else 33

            # === Insert into Event ===
            cur.execute("""
                INSERT INTO "Event" (device_id, timestamp, type, image_path, max_score)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (device_id, timestamp, DB_TYPE, full_image_path, max_score))
            event_id = cur.fetchone()[0]

            # === Insert EventChannels (channels + new_platform) ===
            channel_records = [
                (event_id, name, round(score, 2))
                for name, score in channel_detections + new_platform_detections+ old_platform_detections+youtube_detections
            ]
            if channel_records:
                cur.executemany("""
                    INSERT INTO "EventChannel" (event_id, name, score)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, channel_records)

            # === Insert EventAds ===
            ad_records = [
                (event_id, name, round(score, 2))
                for name, score in ad_detections
            ]
            if ad_records:
                cur.executemany("""
                    INSERT INTO "EventAd" (event_id, name, score)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, ad_records)

            # === Insert EventContent (new_content + old_content) ===
            content_records = [
                (event_id, name, round(score, 2))
                for name, score in new_content_detections + old_content_detections
            ]#+ phash_detections
            if content_records:
                cur.executemany("""
                    INSERT INTO "EventContent" (event_id, name, score)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, content_records)

            conn.commit()
            inserted = 1
            duration = round(time.time() - start_time, 3)

            logger.info(
                f"Inserted event {event_id} for device {device_id} "
                f"(Channels: {len(channel_records)}, Ads: {len(ad_records)}, Content: {len(content_records)}) "
                f"in {duration}s"
            )

 # === Insert EventOCR ===
            # ocr_records = [
            #     (event_id, text, round(score, 2))
            #     for text, score in ocr_detections
            # ]

            # if ocr_records:
            #     cur.executemany("""
            #         INSERT INTO "EventOCR" (event_id, text, score)
            #         VALUES (%s, %s, %s)
            #         ON CONFLICT DO NOTHING
            #     """, ocr_records)

            # conn.commit()
            # inserted = 1
            # duration = round(time.time() - start_time, 3)

            # logger.info(
            #     f"Inserted OCR event {event_id} for device {device_id} "
            #     f"(Text: {len(ocr_records)}) "
            #     f"in {duration}s"
            # )


    except Exception as e:
        logger.error(f"Error inserting detection for device {detection.get('device_id')}: {e}")
        conn.rollback()

    finally:
        conn.close()

    return inserted
