import numpy as np
import math
from collections import deque
from vectorfield import FlowField
import cv2
import os

def test_region_2_sorted_lists():
    # Create a dummy instance of the FlowField class
    pictures_name = 'tunnel_depth.png'
    image = cv2.imread(os.path.join('pictures','input', pictures_name))
    image = cv2.resize(image, (10, 10))

    flow_field = FlowField(image)

    # Test case 1: Empty pixel indices
    pixel_indices = []
    sorted_lists = flow_field.region_2_sorted_lists(pixel_indices)
    assert len(sorted_lists) == 0

    # Test case 2: Single pixel index
    pixel_indices = [(5, 5)]
    sorted_lists = flow_field.region_2_sorted_lists(pixel_indices)
    assert len(sorted_lists) == 1
    assert len(sorted_lists[0]) == 1
    assert sorted_lists[0][0] == (5, 5)

    # Test case 3: Multiple pixel indices
    pixel_indices = [(1, 1), (2, 2), (3, 3), (4, 4)]
    sorted_lists = flow_field.region_2_sorted_lists(pixel_indices)
    assert len(sorted_lists) == 1
    assert len(sorted_lists[0]) == 4
    assert sorted_lists[0] == [(1, 1), (2, 2), (3, 3), (4, 4)]

    # Test case 4: Multiple disconnected regions
    pixel_indices = [(1, 1), (2, 2), (3, 3), (8, 8), (9, 9)]
    sorted_lists = flow_field.region_2_sorted_lists(pixel_indices)
    assert len(sorted_lists) == 2
    assert len(sorted_lists[0]) == 3
    assert sorted_lists[0] == [(1, 1), (2, 2), (3, 3)]
    assert len(sorted_lists[1]) == 2
    assert sorted_lists[1] == [(8, 8), (9, 9)]

    print("All tests passed!")

test_region_2_sorted_lists()