from ultralytics import YOLO
import os, cv2
import json

# settings
MODEL_NAME = "data/models/best3.pt"
MODEL_VERSION = "0v1"


seq = "seq10"
IMAGE_FOLDER_PREFIX = f"/data/local-files?d=images/schody_down/{seq}"

IMAGES_DIR = f"data/images/schody_down/{seq}/"
OUTPUT_DIR = f"data/data_files/schody_down/{seq}/"

try:
    os.mkdir(OUTPUT_DIR)
except Exception as e:
    print(f"Folder istnieje: {e}")


def cls_to_name(cls):
    c = {0: "schody"}
    return c[cls]

global_counter = 0
model = YOLO(MODEL_NAME)
for file in os.listdir(IMAGES_DIR):

    image = cv2.imread(IMAGES_DIR + file)
    results = model.predict(image)

    predictions = []
    score = 0
    counter = 0
    for result in results:
        for box in result.boxes.cpu().numpy():
            if box.conf[0].item() < 0.5:
                continue

            xywh = box.xywh[0]
            w = image.shape[0]
            h = image.shape[1]
            predictions.append(
                {
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": w,
                    "original_height": h,
                    "type": "rectanglelabels",
                    "image_rotation": 0,
                    "value": {
                        "rectanglelabels": [cls_to_name(box.cls[0])],
                        "x": (xywh[0].item() - xywh[2].item()/2) / w * 100,
                        "y": (xywh[1].item() - xywh[3].item()/2) / h * 100,
                        "width": xywh[2].item() / w * 100,
                        "height": xywh[3].item() / h * 100,
                    },
                    "score": box.conf.item(),
                }
            )
            score += box.conf
            counter += 1
            global_counter += 1

    if counter > 0:
        score = (score / counter).item()
    data = {
        "data": {"image": f"{IMAGE_FOLDER_PREFIX}/{file}"},
        "predictions": [
            {
                "model_version": MODEL_VERSION,
                "result": predictions,
                "score": score,
            }
        ],
    }

    f = open(f"{OUTPUT_DIR}/{file}.json", "w")
    f.write(json.dumps(data))
    f.close()

print(f"Licznik etykiet: {global_counter}")
