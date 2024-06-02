from transformers import DetrForObjectDetection, DetrImageProcessor


CHECKPOINT = 'facebook/detr-resnet-50'
MODEL_PATH = './custom-model'

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)

HUB_PATH = "rawhad/detr-gui-component-detection-v0"
image_processor.push_to_hub(HUB_PATH)
model.push_to_hub(HUB_PATH)
