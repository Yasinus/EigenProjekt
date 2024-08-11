
import numpy as np
import cv2
import random
import sys
import os
import matplotlib.pyplot as plt
import random as rand

from depth_estimator import DepthEstimator
from segmentation import Segmentation
from scipy.ndimage import sobel


def get_sort_func(sort_type):
     
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
    return func



def apply_mask(img, mask):
    return (mask[..., np.newaxis] * img).astype(np.uint8)



def mask_image(img, sort_type, threshold=(0,100)):
    channel_image = get_sort_func(sort_type)(img) 
    mask = (channel_image > threshold[0]) & (channel_image < threshold[1])
    reverse_mask = ~mask
 
    return apply_mask(img, mask), apply_mask(img, reverse_mask)



def panoptic_to_applied_masks(img,panoptic_segmentation):
    unique_segments = np.unique(panoptic_segmentation)
    applied_masks = []
    for segment_id in unique_segments:
        binary_mask = (panoptic_segmentation == segment_id).astype(np.uint8)
        applied_masks.append(apply_mask(img, binary_mask))

    return applied_masks



# def apply_masks_img(img, binary_masks):
#     applied_masks = []
#     for mask in binary_masks:
#         applied_masks.append(apply_mask(img, mask))
#     return applied_masks



def apply_mask_imgs(imgs, mask):
    applied_masks = []
    for img in imgs:
        applied_masks.append(apply_mask(img, mask))
    return applied_masks



def get_nonzero_indices(matrix):
    nonzero_indices = np.nonzero(matrix)
    index_combinations = list(zip(*nonzero_indices))
    return index_combinations



def grouping_strategy(pixel_indezes, sort_direction):

    pixel_indez_grouped = []
    if sort_direction == 'horizontal':
        pixel_indezes_array = np.array(sorted(pixel_indezes,key=lambda x: x[0]))
        pixel_indez_grouped = [list(map(tuple, group)) for group in np.split(pixel_indezes_array, np.unique(pixel_indezes_array[:, 0], return_index=True)[1])]
    
    elif sort_direction == 'vertical':
        pixel_indezes_array = np.array(sorted(pixel_indezes,key=lambda x: x[1]))
        pixel_indez_grouped = [list(map(tuple, group)) for group in np.split(pixel_indezes_array, np.unique(pixel_indezes_array[:, 1], return_index=True)[1])]
    else:
        pixel_indez_grouped = []

    return pixel_indez_grouped



def interval_groups_more(pixel_indezes):
    intervalrandom_range = (10, 200)
    pixel_indez_grouped = []
    neighborhood_range = 1

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


def get_gradient_matrix(img):
    sobel_x = sobel(img, axis=0)
    sobel_y = sobel(img, axis=1)
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    #gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    gradient_angle = np.arctan2(sobel_y, sobel_x)

    return gradient_magnitude, gradient_angle



def sort_pixels(img, sort_type, sort_direction, sort_ascend = True):
    output = img.copy()
    channel_image = get_sort_func(sort_type)(img) 
    pixel_index = get_nonzero_indices(channel_image)
    pixel_indez_grouped = grouping_strategy(pixel_index, sort_direction)
    pixel_indez_grouped = interval_groups_more(pixel_indez_grouped)

    for group in pixel_indez_grouped:
        group_sorted = sorted(group, key= lambda x: channel_image[x] , reverse=sort_ascend)
        for i, index in enumerate(group):
            output[index] = img[group_sorted[i]]

    return output



def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():

    pictures_name = 'light.png'
    sort_type = 'value'
    use_depth = True
    use_segmentation = True

    image = cv2.imread(os.path.join('pictures','input', pictures_name))
    #image = cv2.resize(img, (500, 500))
    show_image(image)

    if use_depth:
        depth_estimator = DepthEstimator()
        image_depth = depth_estimator.estimate_depth(image)
        show_image(image_depth)
        #cv2.imwrite(os.path.join('pictures','output', 'depth_' + pictures_name), image_depth)

    if use_segmentation:
        segmentator = Segmentation()
        panoptic_segmentation = segmentator.segment_image(image)
        #show_image(panoptic_segmentation * 255)
        applied_masks = panoptic_to_applied_masks(image,panoptic_segmentation)
        for mask in applied_masks:
            show_image(mask)
    else:
        applied_masks = [image]


    cummulativ_image = np.zeros_like(image)
    for applied_seg_image in applied_masks:
        masked_image,reverse = mask_image(applied_seg_image, sort_type, threshold=(60, 245))
        second_masked_image,reverse = mask_image(reverse, sort_type, threshold=(0, 60))
        show_image(masked_image)
        show_image(second_masked_image)

        sorted_image = sort_pixels(masked_image, sort_type=sort_type, sort_direction='vertical')
        second_sorted_image = sort_pixels(second_masked_image, sort_type=sort_type, sort_direction='horizontal')
        output_image = sorted_image + second_sorted_image + reverse
        cummulativ_image += output_image
        show_image(output_image)

    show_image(cummulativ_image)

    #cv2.imwrite(os.path.join('pictures','output', pictures_name), img)

if __name__ == '__main__':
    main()
