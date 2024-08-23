import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import cv2 
import os
from collections import deque
import concurrent.futures

class FlowField:

    def __init__(self, image, invert = True, weight = 0.3,experiment = False):
        self.graph = {}
        self.inverted_graph = {}
        self.neighbors = None
        self.flow_field = np.zeros((image.shape[0], image.shape[1], 2), dtype=int)
        self.experiment = experiment

        self.image = self.prepocess_image(image, invert)
        self.grad_x, self.grad_y = self.compute_gradient(self.image, weight)
        self.magnitude = np.clip(np.hypot(self.grad_x, self.grad_y), 0, 1) 
        self.angle = self.discretize_angle(np.arctan2(self.grad_y, self.grad_x))    

        self.create_flow_field(self.image)
        self.create_inverted_graph()



    def standardize(self, image):
        return (image - np.mean(image)) / np.std(image)
    


    def standardizefilters(self, x,y):
        concat = np.concatenate((x, y), axis=1)
        standardize_concat = self.standardize(concat)
        x = standardize_concat[:, :x.shape[1]]
        y = standardize_concat[:, x.shape[1]:]
        return x, y
    


    def prepocess_image(self,image, invert = True):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        if invert:
            gray_image = cv2.bitwise_not(gray_image)
        gray_image = self.standardize(gray_image)
        return gray_image
    

    
    def compute_gradient(self, image,weight = 0.3):
        sobel_x = sobel(image, axis=0)  
        sobel_y = sobel(image, axis=1)  
        sobel_x, sobel_y = self.standardizefilters(sobel_x, sobel_y)

        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0) 
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)  
        scharr_x, scharr_y = self.standardizefilters(scharr_x, scharr_y)

        grad_x = scharr_x + weight*  sobel_x
        grad_y = scharr_y + weight*  sobel_y
        grad_x, grad_y = self.standardizefilters(grad_x, grad_y)

        return grad_x, grad_y



    def discretize_angle(self, angle):
        angle = np.round(-np.rad2deg(angle) / 45) * 45 
        if self.experiment:
            angle = (angle +90)
        return angle % 360



    def angle_to_direction(self, angle):
        match angle:
            case 0:
                return 1, 0
            case 45:
                return 1, -1
            case 90:
                return 0, -1
            case 135:
                return -1, -1
            case 180:
                return -1, 0
            case 225:
                return -1, 1
            case 270:
                return 0, 1
            case 315:
                return 1, 1
            case _:
                raise ValueError(f'Invalid angle: {angle}')



    def find_neighbor(self, x, y, angle):
        dx, dy = self.angle_to_direction(angle)
        return x + dx, y + dy



    def create_flow_field(self,image):
        self.flow_field = np.zeros((image.shape[0], image.shape[1], 2), dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                self.flow_field[i, j] = self.find_neighbor(i, j, self.angle[i, j])
        return self.flow_field
 


    def create_inverted_graph(self):
        self.inverted_graph = {}
        for i in range(self.flow_field.shape[0]):
            for j in range(self.flow_field.shape[1]):
                self.inverted_graph[(i, j)] = []

        for i in range(self.flow_field.shape[0]):
            for j in range(self.flow_field.shape[1]):
                x,y = self.flow_field[i, j]
                if 0 <= x < self.flow_field.shape[0] and 0 <= y < self.flow_field.shape[1]:
                    self.inverted_graph[(x, y)].append((i, j))
        return self.inverted_graph



    def bfs_parallel_tree_with_filter(self,tree, root, allowed_nodes):
        visited = set()
        queue = deque([root])
        visited.add(root)
        output = []
        allowed_set = set(allowed_nodes)

        def explore_node(node):
            neighbors = []
            for neighbor in tree[node]:
                if neighbor not in visited and neighbor in allowed_set:
                    neighbors.append(neighbor)
            return neighbors

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while queue:
                level_size = len(queue)
                futures = []
                
                for _ in range(level_size):
                    node = queue.popleft()
                    if node in allowed_set:  
                        output.append(node)
                        futures.append(executor.submit(explore_node, node))
                
                for future in concurrent.futures.as_completed(futures):
                    neighbors = future.result()
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        
        return output[::-1], list(visited)

    
    
    def region_2_sorted_lists(self,pixel_indezes):

        pixel_indezes_copy = pixel_indezes.copy()
        sorted_lists = []
        pixel_indezes_to_remove = []
        masked_image = np.full(self.image.shape, -np.inf)
        for pixel_index in pixel_indezes_copy:
            masked_image[pixel_index] = self.image[pixel_index]
        while pixel_indezes_copy:
            min_pixel_index = np.unravel_index(np.argmax(masked_image, axis=None), masked_image.shape)
        
            sorted_list, pixel_indezes_to_remove = self.bfs_parallel_tree_with_filter(self.inverted_graph, min_pixel_index, pixel_indezes_copy)
            sorted_lists.append(sorted_list)

            for pixel_index in pixel_indezes_to_remove:
                masked_image[pixel_index] = -np.inf
                pixel_indezes_copy.remove(pixel_index)

        return sorted_lists



    def get_nonzero_indices(self,matrix):
        nonzero_indices = np.nonzero(matrix)
        index_combinations = list(zip(*nonzero_indices))
        return index_combinations

    

def main():
    pictures_name = 'tunnel_depth.png'
    image = cv2.imread(os.path.join('pictures','input', pictures_name))
    #image = cv2.resize(image, (100, 100))
    flow_field = FlowField(image, experiment	= True)

    #what = flow_field.region_2_sorted_lists(flow_field.get_nonzero_indices(flow_field.image))
    #print(what)

    # Create a vector field
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    # Plotting the vector field
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', alpha=0.5)
    plt.quiver(x, y, flow_field.grad_x, flow_field.grad_y, flow_field.magnitude, angles='xy', scale_units='xy', scale=1, cmap='viridis')
    plt.title('Vector Field from Image')
    plt.show()
    
    # Plot the discretized angle
    plt.figure(figsize=(10, 10))
    plt.imshow(flow_field.angle, cmap='viridis')
    plt.title('Discretized Angle')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
    