from ultralytics import YOLO, settings
import cv2, os

DATASET         = "daneZRobota"             # folder in dataset 
MODEL_TO_LOAD   = ""          # model to load, if empty load pretrained on coco dataset


IM_SIZE             = 640           # img size
EPOCHS_TO_TRAIN     = 400            # 


# learning parameters 
# https://docs.ultralytics.com/modes/train/#arguments
optimizer           = 'auto'    # choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
lr0                 = 0.01      # initial learning  def=0.01
lrf                 = 0.01      # def=0.01
momentum            = 0.937     # def=0.937


settings.update({'datasets_dir': os.path.abspath("datasets")})
print(settings.get("datasets_dir"))

models_to_train = ["yolov8l", "yolov8m", "yolov8s", "yolov8n"]

for models in models_to_train:
    print("Loading default model")
    model = YOLO(models+'.yaml').load(models+'.pt')    

    results = model.train(
        name = models,
        data = DATASET+'.yaml', 
        epochs = EPOCHS_TO_TRAIN, 
        imgsz = IM_SIZE,
        val = False, 
        optimizer = optimizer, 
        lr0 = lr0, 
        lrf = lrf, 
        momentum = momentum,
        plots = True
        )

    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category
