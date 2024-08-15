
import numpy as np
import cv2
import random
import sys
import os
import matplotlib.pyplot as plt
import random as rand

from depth_estimator import DepthEstimator
from segmentation import Segmentation
from oklab import Oklab
from vectorfield import FlowField
from scipy.ndimage import sobel

class PixelSorter:
    def __init__(self):
        self.flow_field = None


    def get_sort_func(self,sort_type):
        
        match sort_type:
            case 'red':
                func = lambda x: x[:,:,2]
            case 'green':
                func = lambda x: x[:,:,1]
            case 'blue':
                func = lambda x: x[:,:,0]
            case 'hue':
                func = lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2HSV)[:,:,0]
            case 'saturation':
                func = lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2HSV)[:,:,1]
            case 'value':
                func = lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2HSV)[:,:,2]
            case 'okl':
                func = lambda x: Oklab(x).get_lCh()[0]
            case 'okc':
                func = lambda x: Oklab(x).get_lCh()[1]
            case 'okh':
                func = lambda x: Oklab(x).get_lCh()[2]

        return func



    def apply_mask(self,img, mask):
        return (mask[..., np.newaxis] * img).astype(np.uint8)



    def mask_image(self,img, sort_type, threshold=(0,100)):
        channel_image = self.get_sort_func(sort_type)(img) 
        mask = (channel_image >= threshold[0]) & (channel_image <= threshold[1])
        reverse_mask = ~mask
        return mask.astype(np.uint8), reverse_mask.astype(np.uint8)



    def panoptic_to_applied_masks(self,img,panoptic_segmentation):
        unique_segments = np.unique(panoptic_segmentation)
        applied_masks = []
        for segment_id in unique_segments:
            binary_mask = (panoptic_segmentation == segment_id).astype(np.uint8)
            applied_masks.append(self.apply_mask(img, binary_mask))

        return applied_masks



    def apply_mask_imgs(self,imgs, mask):
        applied_masks = []
        for img in imgs:
            applied_masks.append(self.apply_mask(img, mask))
        return applied_masks



    def get_nonzero_indices(self,matrix):
        nonzero_indices = np.nonzero(matrix)
        index_combinations = list(zip(*nonzero_indices))
        return index_combinations



    def grouping_strategy(self,pixel_indezes, sort_direction):

        pixel_indez_grouped = []
        if pixel_indezes == []:
            return pixel_indez_grouped
        if sort_direction == 'horizontal':
            pixel_indezes_array = np.array(sorted(pixel_indezes,key=lambda x: x[0]))
            pixel_indez_grouped = [list(map(tuple, group)) for group in np.split(pixel_indezes_array, np.unique(pixel_indezes_array[:, 0], return_index=True)[1])]
        
        elif sort_direction == 'vertical':
            pixel_indezes_array = np.array(sorted(pixel_indezes,key=lambda x: x[1]))
            pixel_indez_grouped = [list(map(tuple, group)) for group in np.split(pixel_indezes_array, np.unique(pixel_indezes_array[:, 1], return_index=True)[1])]
        elif sort_direction == 'gradient':
            pixel_indez_grouped = self.flow_field.region_2_sorted_lists(pixel_indezes)

        return pixel_indez_grouped 



    def interval_groups_more(self,pixel_indezes, intervalrandom_range): #TODO: take standard deviation into account for interval length	
        intervalrandom_range = (500,500)
        pixel_indez_grouped = []
        neighborhood_range = 2

        for pixel_index_row in pixel_indezes:
            i =0
            while i < len(pixel_index_row):
                random_length = min(random.randint(*intervalrandom_range), len(pixel_index_row) - i)
                neighborhood_group =[]
                for j in range(i, i+random_length):
                    if neighborhood_group != []:
                        is_in_neighborhood = False 
                        for neighborhood_pixel in neighborhood_group: # inefficient but works for checking if pixel is in neighborhood
                            if abs(neighborhood_pixel[0] - pixel_index_row[j][0]) <= neighborhood_range and abs(neighborhood_pixel[1] - pixel_index_row[j][1]) <= neighborhood_range:
                                is_in_neighborhood = True
                                break
                        if not is_in_neighborhood:
                            break
                    neighborhood_group.append(pixel_index_row[j])
                    i += 1

                pixel_indez_grouped.append(neighborhood_group)
        return pixel_indez_grouped



    def get_gradient_matrix(self,img):
        sobel_x = sobel(img, axis=0)
        sobel_y = sobel(img, axis=1)
        gradient_magnitude = np.hypot(sobel_x, sobel_y)
        gradient_angle = np.arctan2(sobel_y, sobel_x)

        return gradient_magnitude, gradient_angle



    def sort_pixels(self,image,mask, sort_type, sort_direction, sort_ascend = True,use_interval_groups = True, intervalrandom_range = (500,500)):
        pixel_index = self.get_nonzero_indices(mask)
        pixel_indez_grouped = self.grouping_strategy(pixel_index, sort_direction)
        output = self.apply_mask(image, mask)
        channel_image = self.get_sort_func(sort_type)(output) 
        if use_interval_groups:
            pixel_indez_grouped = self.interval_groups_more(pixel_indez_grouped, intervalrandom_range)

        for group in pixel_indez_grouped:
            group_sorted = sorted(group, key= lambda x: channel_image[x] , reverse=sort_ascend)
            for i, index in enumerate(group):
                output[index] = image[group_sorted[i]]

        return output



def show_image(img):
    cv2.imshow('Image', img)
    cv2.resizeWindow('Image', 600,600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():

    pictures_name = 'statue.png'
    use_depth = True
    use_segmentation = False

    image = cv2.imread(os.path.join('pictures','input', pictures_name))
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    show_image(image)

    pixelSorter = PixelSorter()

    if use_depth:
        depth_estimator = DepthEstimator()
        image_depth = depth_estimator.estimate_depth(image)
        pixelSorter.flow_field = FlowField(image_depth, experiment = True)

        show_image(image_depth)

    if use_segmentation:
        segmentator = Segmentation()
        panoptic_segmentation = segmentator.segment_image(image)
        #show_image(panoptic_segmentation / np.max(panoptic_segmentation) * 255)
        applied_masks = pixelSorter.panoptic_to_applied_masks(image,panoptic_segmentation)

    else:
        applied_masks = [image]


    cummulativ_image = np.zeros_like(image)
    for applied_seg_image in applied_masks:
        masked_image,reverse = pixelSorter.mask_image(applied_seg_image, 'saturation', threshold=(175,255))
        masked_image,second_reverse = pixelSorter.mask_image(pixelSorter.apply_mask(image, masked_image), 'value', threshold=(105,195))
 
        show_image(pixelSorter.apply_mask(image, masked_image))    

        sorted_image = pixelSorter.sort_pixels(image,
                                                mask=masked_image, 
                                                sort_type='hue', 
                                                sort_direction='vertical', 
                                                sort_ascend=True,
                                                use_interval_groups = True,
                                                intervalrandom_range = (image.shape[0],image.shape[0]))
        sorted_image = pixelSorter.sort_pixels(sorted_image,
                                                mask=masked_image, 
                                                sort_type='value', 
                                                sort_direction='gradient', 
                                                sort_ascend=True,
                                                use_interval_groups = True,
                                                intervalrandom_range = (image.shape[1],image.shape[1]))
        
        output_image = sorted_image + pixelSorter.apply_mask(image, np.clip(reverse + second_reverse,0,1)) 
        cummulativ_image += output_image
        show_image(output_image)

    show_image(cummulativ_image)

    #cv2.imwrite(os.path.join('pictures','output', pictures_name), img)

if __name__ == '__main__':
    main()
