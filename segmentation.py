from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch	

# This class is used to segment an image using the Mask2FormerForUniversalSegmentation model
# The model is pretrained on the COCO dataset
# The segment_image method takes an image as input and returns the segmented image
 
class Segmentation:

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")



    def segment_image(self, image):
        image2 = Image.fromarray(image)
        inputs = self.processor(image2, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[image2.size[::-1]])[0]
        return predictions["segmentation"].cpu().numpy().astype('uint8')
        
 