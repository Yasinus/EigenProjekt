from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image


class DepthEstimator:
    """
    This class is used to estimate the depth of an image using the GLPNForDepthEstimation model
    The model is pretrained on the KITTI dataset
    The estimate_depth method takes an image as input and returns the estimated depth map
    """

    def __init__(self):
        self.processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-kitti")
        self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")



    def estimate_depth(self, image):
        """
        This method estimates the depth of an image
        Args:
            image (np.array): the image for which the depth needs to be estimated
        Returns:
            np.array: the estimated depth map
        """

        image2 = Image.fromarray(image)
        inputs = self.processor(images=image2, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image2.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        return formatted
    
 