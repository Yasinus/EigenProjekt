
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


from flowfield import FlowField
from colorspace import ColorSpace

# This is just an experiment to see how the depth of field effect can be applied to an image
# also it simulates the effect of Chromatic aberration in a lens that is able to shift the colors in the image with different colorspaces
# it ultimately did not look as cool as I thought it would be

class DepthOfField:
    def __init__(self, image, depth_map, focus_distance, aperture_size, use_gradient= False):
        self.image = image
        self.depth_map = depth_map
        self.focus_distance = focus_distance
        self.aperture_size = aperture_size
        self.use_gradient = use_gradient
        if use_gradient:
            flow_field = FlowField(image)
            self.mean_direction = flow_field.angle_to_direction(flow_field.discretize_angle(np.mean(flow_field.angle)))
            

    def apply(self, effect='gaussian', color_space='rgb'):
        output_image = np.zeros_like(self.image, dtype=np.float64)

        for i in range(256):
            mask = cv2.inRange(self.depth_map, i, i) / 255
            effect_size = self.aperture_size * abs(self.focus_distance - (i / 255.0))

            match effect:
                case 'gaussian':
                    kernel_size = int(2 * np.ceil(3 * effect_size) + 1) 
                    effect_image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigmaX=effect_size, sigmaY=effect_size)
                case 'space_shift':
                    effect_size = int(effect_size)
                    c1,c2,c3 = cv2.split(ColorSpace.get_color_space(color_space)(self.image))
                    if self.use_gradient:
                        self.mean_direction = (0,1)
                    c1 = np.roll(c1, self.mean_direction[0]*effect_size, axis=0)
                    c1 = np.roll(c1, self.mean_direction[1]*effect_size, axis=1)

                    c3 = np.roll(c3, -self.mean_direction[0]*effect_size, axis=0)
                    c3 = np.roll(c3, -self.mean_direction[1]*effect_size, axis=1)

                    effect_image = ColorSpace.get_color_space_inverse(color_space)(cv2.merge((c1,c2,c3)))

            output_image = cv2.add(output_image, effect_image * mask[:, :, np.newaxis])
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        return output_image

# Load the image and depth map
image = cv2.imread(os.path.join('pictures','input', 'tunnel.png'))
depth_map = cv2.imread(os.path.join('pictures','input', 'tunnel_depth.png'), cv2.IMREAD_GRAYSCALE)
    
dof = DepthOfField(image, depth_map, 0.5, 10, use_gradient=True) 
output_image = dof.apply(effect='space_shift', color_space='lab')

cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
