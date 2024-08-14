import concurrent.futures
from collections import deque

def bfs_parallel_tree_with_filter(tree, root, allowed_nodes):
    visited = set()
    queue = deque([root])
    visited.add(root)
    output = []

    # Convert allowed_nodes to a set for O(1) lookup
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
                if node in allowed_set:  # Only consider nodes in the allowed_set
                    output.append(node)
                    futures.append(executor.submit(explore_node, node))
            
            for future in concurrent.futures.as_completed(futures):
                neighbors = future.result()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    
    return output, list(allowed_set - visited)

# Example usage
tree = {
    0: [1, 2],
    1: [3, 4],
    2: [],
    3: [5],
    4: [],
    5: []
}

root = 0
allowed_nodes = [0, 1, 2, 3, 4,6]  # Ignoring node 5
bfs_output,set = bfs_parallel_tree_with_filter(tree, root, allowed_nodes)
print("BFS output:", bfs_output)
print(set)
