from ultralytics import YOLO
from chamber_obb.preprocessing import load_and_normalize
from pathlib import Path

# Trainingparameters
NUM_EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 0
# Verschiedene res
IMGSZ = 640
# Variiert imgsz in jedem batch um +/- multiscale
MULTISCALE = 0.5
SEED = 42
DATA_TRAIN = Path("/home/ubuntu/work/workspace/Projekt/src/data.yaml")
SOURCE_TRAIN = Path("/home/ubuntu/work/workspace/image_analysis/roi_export")
CHIPS_PATH = Path("/home/ubuntu/work/workspace/Projekt/chips")

# Modelparameters
MODEL = YOLO("yolo26n-obb.pt")



# Inferenceparameters
SOURCE_INF = Path("/home/ubuntu/work/workspace/Projekt/inference_check")
IMAGES_INF = [load_and_normalize(p) for p in sorted(SOURCE_INF.glob("*.tif*"))]
MODEL_INF = YOLO("runs/obb/train/weights/best.pt")
# Train 1 best run

# Augmentierungsparameter
HSV_V = 0.6 # 0.4 ist standart, belichtungsunterschiede sind glaube nicht so groß bei uns?
# Macht Saturation und Hue sinn bei Graustufen bildern? eher nicht
DEGREES = 180 # Macht denke ich Sinn für OBB und da unsere Kammern ja auch nicht immer grade sind
TRANSLATE = 0.3 # Könnte sinn machen damit die Boxen nicht von der Position immer ähnlich sind?
SCALE = 0.3 # ^
FLIPUD = 0.5 
FLIPLR = 0.5 # 0.5 sowieso standard
MOSAIC = 1 # Hilft komplexe szenen zu verstehen, eher nicht der fall bei uns?
ERASING = 0.5 # 0.4 Standard. Könnte helfen aber auch probleme machen ? Unschlüssig


#Testingparameter
MODEL_TEST = YOLO("runs/obb/train/weights/best.pt")  
MODEL_CHECKUP = YOLO("runs/obb/train/weights/best.pt")
