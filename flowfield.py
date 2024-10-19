import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
import cv2 
import os
from collections import deque
import concurrent.futures



class FlowField:
    """
    This class is used to create a flow field from an image. The flow field is used to create a depth of field effect.
    
    Attributes:
    image: The image from which the flow field is created.
    grad_x: The gradient in the x direction.
    grad_y: The gradient in the y direction.
    magnitude: The magnitude of the gradient.
    angle: The angle of the gradient.
    flow_field: The flow field.
    inverted_graph: The inverted graph of the flow field.

    The flow field precomputes the flow field and the inverted graph of the flow field.
    Only relevant for usage is region_2_sorted_lists, which is used in the PixelSorter class

    """

    def __init__(self, image, invert = True, weight = 0.3):
        self.graph = {}
        self.inverted_graph = {}
        self.neighbors = None
        self.flow_field = np.zeros((image.shape[0], image.shape[1], 2), dtype=int)


        self.image = self.prepocess_image(image, invert)
        self.grad_x, self.grad_y = self.compute_gradient(self.image, weight)
        self.magnitude = np.clip(np.hypot(self.grad_x, self.grad_y), 0, 1) 
        self.angle = self.discretize_angle(np.arctan2(self.grad_y, self.grad_x))    

        self.create_flow_field(self.image)
        self.create_inverted_graph()



    def standardize(self, image):
        return (image - np.mean(image)) / np.std(image)
    


    def standardize_filters(self, x,y):
        concat = np.concatenate((x, y), axis=1)
        standardize_concat = self.standardize(concat)
        x = standardize_concat[:, :x.shape[1]]
        y = standardize_concat[:, x.shape[1]:]
        return x, y
    


    def prepocess_image(self,image, invert = True):
        """
        Preprocess the image by converting it to grayscale and standardizing it.
        
        Args:
        image: The image that is preprocessed.
        invert: If True, the image_values of the one channel image are inverted

        """

        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        if invert:
            gray_image = cv2.bitwise_not(gray_image)
        gray_image = self.standardize(gray_image)
        return gray_image
    

    
    def compute_gradient(self, image,weight = 0.3):
        """
        Compute the gradient of the image using the Scharr operator and the Sobel operator.
        The combination of both looked best in the experiments.
        the gradient is standardized to have a mean of 0 and a standard deviation of 1.
        
        Args:
        image: The image from which the gradient is computed.
        weight: The weight of the Sobel operator in the gradient computation.

        """

        sobel_x = sobel(image, axis=0)  
        sobel_y = sobel(image, axis=1)  
        sobel_x, sobel_y = self.standardize_filters(sobel_x, sobel_y)

        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0) 
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)  
        scharr_x, scharr_y = self.standardize_filters(scharr_x, scharr_y)

        grad_x = scharr_x + weight*  sobel_x
        grad_y = scharr_y + weight*  sobel_y
        grad_x, grad_y = self.standardize_filters(grad_x, grad_y)

        return grad_x, grad_y



    def discretize_angle(self, angle):
        """
        Discretize the angle to 45 degree steps. This allows us to think of the flow field as a graph.
        
        Args:
        angle: The angle that is discretized.
        
        Returns:
        angle: The discretized angle.
        """
        angle = np.round(-np.rad2deg(angle) / 45) * 45 
        return angle % 360



    def angle_to_direction(self, angle):
        """
        Convert an angle to a direction vector.
        This is useful for finding the neighbors of a pixel in the flow field.

        Args:
        angle: The angle that is converted to a direction vector.

        Returns:
        dx, dy: The direction vector of the angle.

        """
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
        """
        Find the neighbor of a pixel in the flow field based on the angle of the pixel.

        Args:
        x: The x coordinate of the pixel.
        y: The y coordinate of the pixel.
        angle: The angle of the pixel.

        Returns:
        x, y: The neighbor of the pixel.
        """

        dx, dy = self.angle_to_direction(angle)
        return x + dx, y + dy



    def create_flow_field(self,image):
        """
        Create the flow field from the image.
        The flow field is a 2D array where each pixel has a direction to its neighbor.
        The direction is based on the gradient of the image.
        The result is tree structure where each pixel has a parent pixel. 

        Args:
        image: The image from which the flow field is created.

        Returns:
        flow_field: The flow field of the image.
            
        """

        self.flow_field = np.zeros((image.shape[0], image.shape[1], 2), dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                self.flow_field[i, j] = self.find_neighbor(i, j, self.angle[i, j])
        return self.flow_field
 


    def create_inverted_graph(self):
        """
        Create the inverted graph of the flow field.
        The inverted graph is a dictionary where the key is a pixel and the value is a list of neighbors of the pixel.

        Returns:
        inverted_graph: The inverted graph of the flow field.
        """

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
        """
        Perform a parallel breadth-first search on a tree with a filter.
        The filter is used to only visit nodes that are in the allowed_nodes set.
        The function is used to sort the pixels of an image based on the flow field.
        The function is parallelized using the ThreadPoolExecutor for faster performance.
        we use dynamic programming to store the visited nodes and the output list of pixel indezes.

        Args:
        tree: The tree on which the breadth-first search is performed.
        root: The root of the tree.
        allowed_nodes: The set of nodes that are allowed to be visited.

        Returns:
        output: The sorted list of pixel indezes.
        visited: The visited nodes.
        """
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

    
    
    def region_2_sorted_lists(self,pixel_indices):
        """
        This function creates lists of pixel indezes based on the flow field.
        Since we have a tree structure, we split the tree into multiple trees on the root level.
        We then traverse each tree in parallel using a breadth-first search.
        The idea is to split the trunks of the tree into branches and then sort the branches in parallel, till we reach the leafs of the tree.
        In the end the branches are only the sorted lists of pixel indezes.
        Args:
        pixel_indices: The pixel indezes that are sorted.

        Returns:
        sorted_lists: The sorted lists of pixel indezes.
        """

        pixel_indices_copy = pixel_indices.copy()
        sorted_lists = []
        pixel_indices_to_remove = []
        masked_image = np.full(self.image.shape, -np.inf)
        for pixel_index in pixel_indices_copy:
            masked_image[pixel_index] = self.image[pixel_index]
        while pixel_indices_copy:
            min_pixel_index = np.unravel_index(np.argmax(masked_image, axis=None), masked_image.shape)
        
            sorted_list, pixel_indices_to_remove = self.bfs_parallel_tree_with_filter(self.inverted_graph, min_pixel_index, pixel_indices_copy)
            sorted_lists.append(sorted_list)

            for pixel_index in pixel_indices_to_remove:
                masked_image[pixel_index] = -np.inf
                pixel_indices_copy.remove(pixel_index)

        return sorted_lists



    

def main():
    # example usage of the FlowField class to create a flow field from an image
    ## as example we use the tunnel_depth.png image is a precalculated depth map of the tunnel.png image
    pictures_name = 'tunnel_depth.png'
    image = cv2.imread(os.path.join('pictures','input', pictures_name))
    #image = cv2.resize(image, (100, 100))
    flow_field = FlowField(image)



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
    