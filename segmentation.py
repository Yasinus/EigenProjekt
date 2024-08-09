from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch	

# processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

class Segmentation:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

    def segment_image(self, image):
        image2 = Image.fromarray(image)
        inputs = self.processor(image2, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        mask_queries_logits = outputs.outputs.masks_queries_logits
        mask_queries = mask_queries_logits.argmax(1)
        mask_queries = mask_queries.cpu().numpy()
        mask_queries = mask_queries.squeeze()
        return mask_queries

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)    
segmenter = Segmentation()
mask = segmenter.segment_image(image)
mask = Image.fromarray(mask)
mask.show()
