from oklab import Oklab
import cv2
from depth_estimator import DepthEstimator



class ColorSpace:
    """

    This class contains methods that are used to sort images by their color channels and convert images to different color spaces.
    It is a library of functions and is not meant to be instantiated.

    """

    @staticmethod
    def get_sort_func(sort_type):
        """
        This method returns a function that extracts the channel of an image that we want to sort by

        Args:
            sort_type (str): the channel we want to sort by
        """
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
            case 'depth': #special case for depth for experimental purposes
                func = lambda x: DepthEstimator().estimate_depth(x)
        return func
    

    
    @staticmethod
    def get_color_space(color_space):
        """
        This method returns a function that converts an image to a different color space

        Args:
            color_space (str): the color space we want to convert to
        """

        match color_space:
            case 'rgb':
                func = lambda x: x
            case 'hsv':
                func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
            case 'lab':
                func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2Lab)
            case 'oklab':
                func = lambda x: Oklab(x).get_oklab()
            case 'oklch':
                func = lambda x: Oklab(x).get_lCh()
            case 'yCrCb':
                func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
        return func
    
    @staticmethod
    def get_color_space_inverse(color_space):
        """
        This method returns a function that converts an image from a different color space to RGB

        Args:
            color_space (str): the color space we want to convert from
        """
        match color_space:
            case 'rgb':
                func = lambda x: x
            case 'hsv':
                func = lambda x: cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
            case 'lab':
                func = lambda x: cv2.cvtColor(x, cv2.COLOR_Lab2BGR)
            case 'oklab':
                func = lambda x: Oklab(x).get_rgb() #TODO: Implement this
            case 'oklch':
                func = lambda x: Oklab(x).get_rgb()
            case 'yCrCb':
                func = lambda x: cv2.cvtColor(x, cv2.COLOR_YCrCb2BGR)
        return func
