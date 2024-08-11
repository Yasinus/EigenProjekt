import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter, laplace, prewitt 
import cv2 
import os



def standardize(image):
    return (image - np.mean(image)) / np.std(image)



pictures_name = 'tunnel_depth.png'

image = cv2.imread(os.path.join('pictures','input', pictures_name))
image = cv2.resize(image, (50, 50))
#image = image[:112, :123]

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
invert_image = cv2.bitwise_not(gray_image)
gray_image = invert_image

#gray_image = gaussian_filter(gray_image, sigma=1.5)
gray_image = standardize(gray_image)


# sobel_x = sobel(gray_image, axis=0)  # Horizontal gradient
# sobel_y = sobel(gray_image, axis=1)  # Vertical gradient
# sobel_x = prewitt(gray_image, axis=0)  # Horizontal gradient
# sobel_y = prewitt(gray_image, axis=1)  # Vertical gradient

sobel_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
sobel_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)

sobel_concat = np.concatenate((sobel_x, sobel_y), axis=1)
standardize_sobel = standardize(sobel_concat)
sobel_x = standardize_sobel[:, :sobel_x.shape[1]]
sobel_y = standardize_sobel[:, sobel_x.shape[1]:]



# # Starting point for steepest descent (e.g., center of the image)
# x, y = gray_image.shape[1] // 2, gray_image.shape[0] // 2

# # Perform steepest descent to find the darkest point
# steps = 100  # Number of steps to take
# path = [(x, y)]  # Track the path

# for _ in range(steps):
#     # Get the gradient at the current position
#     gx = sobel_x[y, x]
#     gy = sobel_y[y, x]
    
#     # Move in the direction opposite to the gradient (steepest descent)
#     x = x - int(np.sign(gx))
#     y = y - int(np.sign(gy))
    
#     # Ensure we stay within image bounds
#     x = np.clip(x, 0, gray_image.shape[1] - 1)
#     y = np.clip(y, 0, gray_image.shape[0] - 1)
    
#     path.append((x, y))
    
#     # Stop if the gradient is near zero (local minimum)
#     if np.hypot(gx, gy) < 1e-5:
#         break

# # Plot the path taken during steepest descent
# plt.figure(figsize=(10, 10))
# plt.imshow(gray_image, cmap='gray')
# path = np.array(path)
# plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=5)  # Red path
# plt.title('Enhanced Steepest Descent Path to the Darkest Point')
# plt.show()

# plt.figure(figsize=(10, 10))
# plt.imshow(sobel_x, cmap='gray')
# plt.title('Horizontal Gradient (Sobel Operator)')
# plt.show()

# plt.figure(figsize=(10, 10))	
# plt.imshow(sobel_y, cmap='gray')
# plt.title('Vertical Gradient (Sobel Operator)')
# plt.show()


# Compute magnitude and direction
magnitude = np.hypot(sobel_x, sobel_y)  # Magnitude of gradient
angle = np.arctan2(sobel_y, sobel_x)    # Angle of gradient

# Create a vector field
x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

# Plotting the vector field
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray', alpha=0.5)
plt.quiver(x, y, sobel_x, sobel_y, magnitude, angles='xy', scale_units='xy', scale=1, cmap='jet')
plt.title('Vector Field from Image')
plt.show()
 