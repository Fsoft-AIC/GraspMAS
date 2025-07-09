from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon



def rotated_rect_to_polygon(x, y, w, h, angle):
    """Convert a rotated rectangle to a polygon."""
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    corners = np.array([
        [-w / 2, -h / 2],
        [ w / 2, -h / 2],
        [ w / 2,  h / 2],
        [-w / 2,  h / 2]
    ])
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    return Polygon(rotated_corners)

def calculate_iou_and_angle(rect1, rect2, dataset_type='GraspAnything'):
    """Calculate IoU and angle between two rotated rectangles."""
    x1, y1, w1, h1, angle1 = rect1
    if dataset_type == 'GraspAnything':
        _, x2, y2, w2, h2, angle2 = rect2
    else: 
        x2, y2, w2, h2, angle2, _ = rect2
    poly1 = rotated_rect_to_polygon(x1, y1, w1, h1, angle1)
    poly2 = rotated_rect_to_polygon(x2, y2, w2, h2, angle2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union if union != 0 else 0
    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, 180 - angle_diff)
    return iou, angle_diff

def eval_grasp(predict, ground_truth, iou_threshold=0.25, angle_threshold=30):
    max_iou=0
    for gt in ground_truth:
        iou_score, angle_diff = calculate_iou_and_angle(predict, gt.tolist())
        if angle_diff > angle_threshold:
            continue
        max_iou = max(max_iou, iou_score)
    # print(max_iou)
    if max_iou < iou_threshold:
        return False
    return True

def visualize_grasp_pose(image, grasp_pose):
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
    output_path = '/LOCAL2/anguyen/faic/quang/viper_duality/imgs/grasp_pose_visualization.png'
    plt.savefig(output_path)
    return Path(output_path)