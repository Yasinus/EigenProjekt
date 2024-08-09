
import numpy as np
import cv2
import random
import sys
import os
import matplotlib.pyplot as plt
import random as rand

from depth_estimator import DepthEstimator



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


def mask_image(img, sort_type, threshold=(0,100)):
    channel_image = get_sort_func(sort_type)(img) 
    mask = (channel_image > threshold[0]) & (channel_image < threshold[1])
    reverse_mask = ~mask
    channel_image = (mask[..., np.newaxis] * img).astype(np.uint8)
    reverse_channel_image = (reverse_mask[..., np.newaxis] * img).astype(np.uint8)
    return channel_image, reverse_channel_image


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



 
    

def main():

    # Read the image
    img = cv2.imread(os.path.join('pictures','input', 'light.png'))
    #img = cv2.resize(img, (500, 500))
    
    depth_estimator = DepthEstimator()
    depth = depth_estimator.estimate_depth(img)
    cv2.imshow('Image', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    sort_type = 'value'
    img,reverse = mask_image(img, sort_type, threshold=(60, 245))
    second,reverse = mask_image(reverse, sort_type, threshold=(0, 60))

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #img = sort_pixels(img, sort_type=sort_type, sort_direction='horizontal')
    img = sort_pixels(img, sort_type=sort_type, sort_direction='vertical')
    second = sort_pixels(second, sort_type=sort_type, sort_direction='horizontal')
    img += reverse
    img += second
    
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite(os.path.join('pictures','output', 'light.png'), img)

if __name__ == '__main__':
    main()
