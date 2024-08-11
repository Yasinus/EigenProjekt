import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx

# Define the radial vector field
def vector_field(x, y):
    u = x  # Flow in the x direction
    v = y  # Flow in the y direction
    return np.array([u, v])

# Calculate the magnitude of the vector field at a point
def vector_magnitude(x, y):
    vec = vector_field(x, y)
    return np.linalg.norm(vec)

# Find the point with the highest vector magnitude
def find_highest_magnitude_point(grid_x, grid_y):
    magnitudes = np.array([[vector_magnitude(x, y) for x in grid_x] for y in grid_y])
    max_idx = np.unravel_index(np.argmax(magnitudes), magnitudes.shape)
    return grid_x[max_idx[1]], grid_y[max_idx[0]]

# Trace flow line starting from (x0, y0)
def trace_flow_line(x0, y0, t):
    def flow_derivative(pos, t):
        x, y = pos
        return vector_field(x, y)
    
    initial_pos = [x0, y0]
    flow_line = odeint(flow_derivative, initial_pos, t)
    return flow_line

# Generate a grid for the vector field
grid_x = np.linspace(-2, 2, 50)
grid_y = np.linspace(-2, 2, 50)

# Find the point with the highest magnitude
root_x, root_y = find_highest_magnitude_point(grid_x, grid_y)

# Time steps for integration
t = np.linspace(0, 1, 100)

# Trace the flow line starting from the highest magnitude point
flow_line = trace_flow_line(root_x, root_y, t)

# Create a tree structure using NetworkX
G = nx.DiGraph()

# Populate the tree with flow line
prev_node = (root_x, root_y)
G.add_node(prev_node)  # Add the root node
for point in flow_line[1:]:
    node = tuple(point)
    G.add_edge(prev_node, node)
    prev_node = node

# Plot the vector field
x, y = np.meshgrid(grid_x, grid_y)
u, v = vector_field(x, y)
plt.quiver(x, y, u, v, color='lightgray')

# Plot the flow line
plt.plot(flow_line[:, 0], flow_line[:, 1], lw=2, color='blue')

# Draw the tree structure
pos = {node: node for node in G.nodes()}
nx.draw(G, pos, node_size=50, node_color="red", edge_color="blue", with_labels=False)

plt.title("Flow from the Highest Magnitude Point in Radial Field")
plt.show()

