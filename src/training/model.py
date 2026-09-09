import config


model = config.MODEL

def inference():
    model = config.MODEL_INF
    images = config.IMAGES_INF
    results = model(images, save=True, project="inference_results", name="run1", conf=0.8)

    for result in results:
        print(result)


def train():
    model.train(
        data=config.DATA_TRAIN, 
        epochs=config.NUM_EPOCHS,
        imgsz=config.IMGSZ,
        multi_scale=config.MULTISCALE,
        task="obb",
        batch=config.BATCH_SIZE,
        hsv_v=config.HSV_V,
        degrees=config.DEGREES,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        flipud=config.FLIPUD,
        fliplr=config.FLIPLR,
        mosaic=config.MOSAIC, # Eventuell rausnehmen
        erasing=config.ERASING # Eventuell rausnehmen
        )


    model.predict("dataset/images/val", save=True, conf=0.25)
    # Maybe run a whole test loop after train and validation

def test():
    model = config.MODEL_TEST

    metrics = model.val(
        data=config.DATA_TRAIN,   
        split="test",              # explizit test als split angeben weil er sonst val auswertet
    )

    print(f"mAP50: {metrics.box.map50}")       # mAP50
    print(f"mAP50-95: {metrics.box.map}")         # mAP50-95
    print(f"Precision: {metrics.box.p}")           # Precision
    print(f"Recall: {metrics.box.r}")           # Recall

def checkUp():
    model = config.MODEL_CHECKUP
    model.predict("dataset/images/train", save=True, conf=0.25)