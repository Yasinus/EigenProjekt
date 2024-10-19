
import numpy as np
import cv2
import random
import os


from colorspace import ColorSpace
from mask import Mask
from flowfield import FlowField



class PixelSorter:
    """
    PixelSorter class is used to sort the pixels of an image 
    The most important method is sort_pixels which sorts the pixels of an image based on a mask and a sort type.
    It needs a mask to define the all the pixels of the pixels that should be sorted.
    We are able to sort the pixels in a horizontal, vertical or from the flow field direction.
    This is basically how the array of pixels, which should be sorted, is created.
    The flowfield has to be precomputed for the entire image and it is not done in this class.

    The pixels can be sorted in ascending or descending order. 
    The value of the pixels is determined by the sort type, which is either a color channel or the depth of the pixel.
    See ColorSpace class for more information about the sort types.

    We can also split the pixels into even smaller groups and sort them individually. 
    This can increase the randomness of the sorting and also increase the performance.

    """
    
    def __init__(self): 
        pass


    def get_nonzero_indices(self,matrix):
        """
        This method returns the indices of the nonzero elements of a matrix
        Args:
            matrix (np.array): the matrix we want to get the indices from
        
        Returns:
            list: a list of tuples containing the indices of the nonzero elements
        """

        nonzero_indices = np.nonzero(matrix)
        index_combinations = list(zip(*nonzero_indices))
        return index_combinations



    def grouping_strategy(self,pixel_indices, sort_direction, flow_field = None):
        """
        This method groups the pixel indices based on the sort direction
        Args:
            pixel_indices (list): a list of tuples containing the pixel indices
            sort_direction (str): the direction we want to sort the pixels in
            flow_field (FlowField): the flow field of the entire image: used for only flowfield sorting
        Returns:
            list: a list of lists containing the grouped pixel indices
        """

        pixel_indices_grouped = []
        if pixel_indices == []:
            return pixel_indices_grouped
        
        match sort_direction:
            case 'horizontal':
                pixel_indices_array = np.array(sorted(pixel_indices,key=lambda x: x[0]))
                pixel_indices_grouped = [list(map(tuple, group)) for group in np.split(pixel_indices_array, np.unique(pixel_indices_array[:, 0], return_index=True)[1])]

            case 'vertical':
                pixel_indices_array = np.array(sorted(pixel_indices,key=lambda x: x[1]))
                pixel_indices_grouped = [list(map(tuple, group)) for group in np.split(pixel_indices_array, np.unique(pixel_indices_array[:, 1], return_index=True)[1])]
        
            case 'flowfield':
                if flow_field is None:
                    raise ValueError('Flow field of the entire image is not set') #we need the precomputed flow field for high performance
                pixel_indices_grouped = flow_field.region_2_sorted_lists(pixel_indices)

        return pixel_indices_grouped 



    def split_interval_groups_more(self,pixel_indices, interval_random_range): #TODO: take standard deviation into account for interval length	
        """	
        This method splits the pixel indices into smaller groups
        Args:
            pixel_indices (list): a list of lists containing the pixel indices
            interval_random_range (tuple): the range of the random interval length
        Returns:
            list: a list of lists containing the grouped pixel indices
        """
        
        pixel_indices_grouped = []
        neighborhood_range = 2

        for pixel_index_row in pixel_indices:
            i =0
            while i < len(pixel_index_row):
                random_length = min(random.randint(*interval_random_range), len(pixel_index_row) - i)
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

                pixel_indices_grouped.append(neighborhood_group)
        return pixel_indices_grouped



    def sort_pixels(self,image,mask, sort_type, sort_direction, sort_ascend = True,split_interval_groups = True, interval_random_range = (500,500),flow_field = None):
        """
        This method sorts the pixels of an image based on a mask and a sort type

        Args:
            image (np.array): the image we want to sort
            mask (np.array): the mask that defines the pixels we want to sort
            sort_type (str): the type of the sort we want to use
            sort_direction (str): the direction we want to sort the pixels in
            sort_ascend (bool): if the pixels should be sorted in ascending order
            split_interval_groups (bool): if the pixels should be split into smaller groups
            interval_random_range (tuple): the range of the random interval length
            flow_field (FlowField): the flow field of the entire image: used for only flowfield sorting
        
        Returns:
            np.array: the sorted image
        """
        
        pixel_indices = self.get_nonzero_indices(mask) #get the pixel indezes of the mask
        pixel_indices_grouped = self.grouping_strategy(pixel_indices, sort_direction, flow_field) #group the pixel indezes based on the sort direction 
        output_image = Mask().apply_mask(image, mask)
        channel_image = ColorSpace.get_sort_func(sort_type)(output_image) #the channel we want to sort by

        if split_interval_groups:
            pixel_indices_grouped = self.split_interval_groups_more(pixel_indices_grouped, interval_random_range)

        for group in pixel_indices_grouped:
            group_sorted = sorted(group, key= lambda x: channel_image[x] , reverse=sort_ascend)
            for i, index in enumerate(group):
                output_image[index] = image[group_sorted[i]]

        return output_image


