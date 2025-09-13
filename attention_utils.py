import numpy as np
from scipy.spatial import distance as dist
import cv2

# Eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Head pose estimation
def get_head_pose(shape, cam_matrix, dist_coeffs):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye
        (shape.part(45).x, shape.part(45).y),     # Right eye
        (shape.part(48).x, shape.part(48).y),     # Left mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye
        (225.0, 170.0, -135.0),      # Right eye
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rotation_vector, translation_vector
