
"""
here are created and stored the functions to be used
"""

import cv2
import numpy as np
import face_recognition
import argparse
from skimage import color
from numpy.linalg import norm
import math

#--------------#
#| FACE COLOR |#
#--------------#
def rgb_to_lab(rgb):
    """ 
    Description
    ---------
    Vabs
    ---------
    Returns
    """
    rgb_norm = np.array(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb_norm.reshape(1,1,3)).reshape(3,)
    return lab

def lab_to_rgb(lab):
    """ 
    Description
    ---------
    Vabs
    ---------
    Returns
    """
    rgb_norm = color.lab2rgb(lab.reshape(1,1,3)).reshap(3,)
    rgb = np.clip(rgb_norm * 255.0, 0, 255).astype(np.unit8)
    return rgb

def cosine_similarity(a, b):
    """ 
    Description
    ---------
    Vabs
    ---------
    Returns
    """
    a = a.flatten()
    b = b.flatten()
    return np.dot(a,b) / (norm(a)*norm(b) + 1e-10)

def get_face_embedding_fr(image, model='small'):
    """ 
    Description
        Returns the first face encoding found (128-d vector) using face_recognition (dlib).
    ---------
    Vabs
        image: 
        model: 'small' or 'large' for landmark/model sizes
    ---------
    Returns
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog') # if GPU/dlib-cnn available -> model='cnn' 
    if len(boxes) == 0:
        return None, None
    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings[0], boxes[0] # encoding and box

def skin_mask_from_landmarks(image, face_location):
    """ 
    Description
        Create a mask for face-skin region using face_landmarks
    ---------
    Vabs
        image:
        face_location: (top, right, bottom, left) from face_recognition
    ---------
    Returns
        binary mask (H,W) where skin region ~1
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BGR2RGB)
    landmarks_list = face_recognition.face_landmarks(rgb, [face_location])
    if not landmarks_list:
        h, w = image.shape[:2]
        return np.zeros((h,w), dttype=np.unit8)
    lm = landmarks_list[0]

    h, w = image.shape[:2]
    mask = np.zeros((h,w), dtype=np.unit8)

    chin = lm.get('chin', [])
    left_eyebrow = lm.get('left_eyebrow', [])
    right_eyebrow = lm.get('right_eyebrow', [])

    poly = []
    poly.extend(chin)

    poly_np = np.array(poly, dtype=np.it32)
    cv2.fillConvexPoly(mask, poly_np, 255)

    #remove eyes and mouth
    for key in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip', 'nose_tip', 'nose_bridge']:
        pts = lm.get(key, [])
        if pts:
            cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 0)

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (21,21), 0)
    # normalize to 0..1
    mask_f = (mask.astype(np.float32) / 255.0)
    return mask_f  # float mask 0..1