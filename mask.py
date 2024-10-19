import numpy as np
from segmentation import Segmentation
from colorspace import ColorSpace

"""
Mask class is used to create masks for an image
A semantic segmentation model can be used to segment the image
"""

class Mask:

    def __init__(self):
        pass



    def apply_mask(self,img, mask):
        """
        This method applies a mask to an image. The mask is a binary image where the pixels are either 0 or 1
        The mask is multiplied with the image to get the masked image, 
        meaning that the pixels of the image that are 0 in the mask will be 0 in the masked image

        Args:
            img (np.array): the image we want to apply the mask to
            mask (np.array): the mask we want to apply

        Returns:
            np.array: the masked image"""
        return (mask[..., np.newaxis] * img).astype(np.uint8)



    def mask_image(self,img, sort_type, threshold=(0,100)):
        """
        This method creates a mask for an image.
        The mask is created by checking if the color channel is within the threshold

        Args:
            img (np.array): the image we want to create the mask for
            sort_type (str): the color channel we want to sort by. See ColorSpace class for more information
            threshold (tuple): the thresholds for the color channel represented as a tuple
        
        Returns:
            np.array: the mask

        """
        
        channel_image = ColorSpace.get_sort_func(sort_type)(img) 
        mask = (channel_image >= threshold[0]) & (channel_image <= threshold[1])
        reverse_mask = ~mask
        return mask.astype(np.uint8), reverse_mask.astype(np.uint8)



    def panoptic_to_applied_masks(self,img,panoptic_segmentation):
        """
        This method creates masks for each segment in the panoptic segmentation
        The masks are applied to the image

        Args:
            img (np.array): the image we want to apply the masks to
            panoptic_segmentation (np.array): the panoptic segmentation of the image
        
        Returns:
            list: a list of the masked images
        """

        unique_segments = np.unique(panoptic_segmentation)
        applied_masks = []
        for segment_id in unique_segments:
            binary_mask = (panoptic_segmentation == segment_id).astype(np.uint8)
            applied_masks.append(self.apply_mask(img, binary_mask))

        return applied_masks



    def apply_mask_imgs(self,imgs, mask):
        """
        This method applies a mask to a list of images

        Args:
            imgs (list): a list of images
            mask (np.array): the mask we want to apply
        
        Returns:
            list: a list of the masked images
        """

        applied_masks = []
        for img in imgs:
            applied_masks.append(self.apply_mask(img, mask))
        return applied_masks



    def segment_image(self, image):
        """
        This method segments an image using the Mask2FormerForUniversalSegmentation model
        The model is pretrained on the COCO dataset

        Args:
            image (np.array): the image we want to segment

        Returns:
            np.array: the segmented images as a list of masked images
        """
        return self.panoptic_to_applied_masks(image,Segmentation().segment_image(image))