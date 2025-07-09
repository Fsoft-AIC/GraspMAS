import cv2
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import os

def visualize_grasp_pose(image, grasp_pose, save_folder='imgs/'):
    """
    Visualize the grasp pose on the image.

    Parameters:
    - image: The input image as a numpy array.
    - grasp_pose: A tuple (quality, x, y, w, h, angle) representing the grasp pose.
        - x, y: Center of the rectangle.
        - w, h: Width and height of the rectangle.
        - angle: Rotation angle of the rectangle in degrees.

    Returns:
    """
    point_color1 = (255, 255, 0)  # BGR  
    point_color2 = (255, 0, 255)  # BGR
    thickness = 2
    lineType = 4
    x, y, w, h, angle = grasp_pose[1:]
    center = (x, y)
    size = (w, h)
    box = cv2.boxPoints((center, size, angle))
    box = np.int64(box)
    plt.figure(figsize=(5,5))
    cv2.line(image, box[0], box[3], point_color1, thickness, lineType)
    cv2.line(image, box[3], box[2], point_color2, thickness, lineType)
    cv2.line(image, box[2], box[1], point_color1, thickness, lineType)
    cv2.line(image, box[1], box[0], point_color2, thickness, lineType)
    plt.imshow(image)
    plt.axis('off')
    os.makedirs(save_folder, exist_ok=True)
    output_path =  os.path.join(save_folder, f"grasp_pose_visualization.png")
    plt.savefig(output_path)
    return Path(output_path)