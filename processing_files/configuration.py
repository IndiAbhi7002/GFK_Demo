# Configuration file for the image processing application
import os
import logging
import numpy as np
# === S3 Configuration file for the image processing application ===
S3_BUCKET = "indi-test-img-processing"
S3_PREFIX = "GFK_Demo/"
S3_LOG_KEY = "GFK_Demo_Output/Logs/demo_image_processing.log"
S3_RECOGNIZED_PREFIX = "GFK_Demo_Output/recognized_frames/"  # Folder for recognized images
S3_UNRECOGNIZED_PREFIX = "GFK_Demo_Output/unrecognized_frames/"  # Folder for unrecognized images
S3_BASE_URL = "https://indi-test-img-processing.s3.ap-south-1.amazonaws.com/"



old_model_content_path = "models/yolo/audit_content_best.pt"
platform_logo_model_path = "models/yolo/audit_logo_model.pt"
new_model_blg_demo_path = "models/yolo/new_best_blg_demo_ep65.pt"  # Path to the unified model
youtube_model_path = "models/youtube/youtube_best.pt"
# nepal_sport_model_path = "models/nepal_sport/nepal_sport_best.pt"
phash_dataset_path = r"models/Phash/25FPS_ott_phash_dataset.npy"

OUTPUT_IMAGE_SIZE = (640, 640)   # (width, height)
OUTPUT_IMAGE_QUALITY = 30


OLD_MODEL_TV_OTT_YT_LIST= ['Zee5','Sony_liv','Sun_Nxt','Hoichoi','JioHotstar','IshowSpeed','MxPlayer','Technical_Guruji','AmazonPrime','Netflix','Erosnow','ABPNews','DDNational','IndiaTV','NDTV','NewsIndia','SansadTV','TImesNowNB','RepublicWorld','ZeeNews','Saregama','Sony_music','Tips_Official','T-Series','AdityaMusic','AnandAudio','ZeeMusic','Dangal','SonySab','Star_pravah','StarPlus', 'Goldmines','Colors','ZeeMarathi','ZeeTV','SunMarathi','Star_Sports1_Hindi', 'TV9_Marathi', 'ALTT','MTV']

OLD_MODEL_CONTENT_LIST= ['Crime_Beat','Khoj','Murshid','Pyaar_Testing ','Vitromates','BadaNaamKarenge','Celebrity_MasterChef_India','Cubicles','Shark_Tank_India','Waking_of_Nation','IshqJabaria','Julali_Gaath_Ga','Nandinika_Pratishodh',
 'Sawali_Hoin_Sukhachi','Shambhavi','Metro_Park','Aisa_Waisa_Pyaar','Hindmata','Date_Gone_Wrong','Modi','Dainee','Bishohori','Eken_Babu ','Nikhoj',
 'Feludar_Goyendagiri','DareDevil','The_Pitt','The_Secret_Of_Shiledars','Laughter_Chefs_Unlimited_Entertainment','Virat@18','The_Wheel_Of_Time','Reacher',
 'House_Of_David','Ziddi_Girls','Aashram ','School_Friends','PyarKa_Professor','Chidiya_Udd','Smile_To_Life','American_Murder_Gabby_Petito','Cassandra ','Black_Warrant','Dabba_Cartel','Squid_Game',
]#'Lust_Party','Utha_Patak','Jawani_Janeman','Namkeen_Kisse','DhakDhak'

YT_MODEL_LIST = ["YouTube"]

OLD_MODEL_EXCLUDE_LIST =['Dettol','5star','Asianpaints','Glow&Lovely','Goodknight','Zee5','Sony_liv','Sun_Nxt','Hoichoi','JioHotstar','IshowSpeed','MxPlayer','Technical_Guruji','AmazonPrime','Netflix','Erosnow','ABPNews','DDNational','IndiaTV','NDTV','NewsIndia','SansadTV','TImesNowNB','RepublicWorld','ZeeNews','Saregama','Sony_music','Tips_Official','T-Series','AdityaMusic','AnandAudio','ZeeMusic','Dangal','SonySab','Star_pravah','StarPlus', 'Goldmines','Colors','ZeeMarathi','ZeeTV','SunMarathi','Star_Sports1_Hindi', 'TV9_Marathi', 'ALTT','MTV']

PLATFORM_LIST = ['Amazon Prime','Zee5','Netflix','Amazon Originals']

CONTENT_LIST = ['Panchayat','Panchayat S2 E1','Panchayat S2 E2','Panchayat S2 E3','Panchayat S2 E4','Panchayat S2 E5','Panchayat S2 E6','Panchayat S2 E7','Panchayat S2 E8','Panchayat S1 E1','Panchayat S1 E2','Panchayat S1 E3','Panchayat S1 E4','Panchayat S3 E1',
'Panchayat S3 E2','Panchayat S3 E3','Panchayat S3 E4','Panchayat S3 E5','Panchayat S3 E6','Panchayat S3 E7','Panchayat S3 E8','Panchayat S4 E5','Panchayat S4 E6','Panchayat S4 E7','Panchayat S4 E8','Panchayat S4 E1',
'Panchayat S4 E2','Panchayat S4 E3','Panchayat S4 E4','Panchayat S1 E5','Panchayat S1 E6','Panchayat S1 E7','Panchayat S1 E8','Khoj Parchaiyo Ke Uss Paar','Khoj S1 E1','Khoj S1 E2','Khoj S1 E3','Khoj S1 E4','Khoj S1 E5','Khoj S1 E6','Khoj S1 E7',
'Bhool Chuk Maaf','Kadak Singh','Squid Game','Squid Game S3 E1','Squid Game S3 E2','Squid Game S3 E3','Squid Game S3 E4','Squid Game S3 E5','Squid Game S3 E6'
]

ADS_LIST = ['Cadbury Celebrations','Kelloggs Chocos','Policy Bazaar','Taco Bell','Apple Iphone 16 AI','Coach','Dabur honey','Dabur Red Paste','HP','Intel','Loreal Paris','Macho Hint','Mahindra','Nescafe Gold','Tata Harrier EV','WhatsApp','Kohler','Kelloggs','Scorpio N','Nescafe','Casting Creme Gloss'
]#'Dabur',,'Dabur Chyawanprash','Hajmola','Lite Horlicks',

# === Database Configuration ===
DB_HOST = "snapshot140825.c960kiumy09x.ap-south-1.rds.amazonaws.com"
DB_NAME = "test01"
DB_USER = "postgres"
DB_PASSWORD = "inditronics"
DB_PORT = 5432

DATA_BASE= "postgres" # (MQTT,postgres,Local MongoDB)

#=== MongoDB Connecion Strign ===
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017/"  # Local MongoDB connection string

# Database constants
DB_TYPE = 29  # Fixed TYPE value as per your requirement

# MQTT setup with certificates
MQTT_BROKER = "a3uoz4wfsx2nz3-ats.iot.ap-south-1.amazonaws.com"  # Change this to your broker's hostname
MQTT_PORT = 8883  # Default secure port for MQTT
MQTT_TOPIC = "apm/server"

CA_CERT_PATH = 'processing_files/root-CA.crt'  # Replace with the path to your CA certificate
CLIENT_CERT_PATH = "processing_files/test.cert.pem.crt"  # Replace with the path to your client certificate (optional)
CLIENT_KEY_PATH = 'processing_files/test.private.pem.key'  # Replace with the path to your client private key (optional)

# Setup logging
log_file = S3_LOG_KEY
os.makedirs(os.path.dirname(log_file), exist_ok=True)

if not os.path.exists(log_file):
    open(log_file, "w").close()

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a",
    force=True
)
logger = logging.getLogger()
