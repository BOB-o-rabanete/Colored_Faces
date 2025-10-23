
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

def skin_mask_from_landmarks(image, face_location) -> float : 
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

def change_skin_color(image_bgr, face_location, target_rgb=(0,128,251), strenght=0.85):
    """ 
    Description
        Change the skin region color toward target_rgb
    ---------
    Vabs
        image_bgr: input OpenCV BGR image
        face_location: (top,right,bottom,left)
        target_rgb: tuple (R,G,B) target color to push skin towards
        strength: [0,1] how strongly to push skin pixels toward target color in LAB space
    ---------
    Returns
        new BGR image (uint8)
    """
    img = image_bgr.copy().astype(np.float32)
    mask = skin_mask_from_landmarks(image_bgr, face_location)
    if mask.sum() > 10:
        top, right, bottom, left = face_location
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        mask[top:bottom, left:right] = 1.0

    rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.floaat32)/255.0
    lab = color.rgb2lab(rgb)

    target_lab = rgb_to_lab(target_rgb)

    # For pixels in mask, move their LAB toward target_lab by strength
    preserve_lightness = True #keep facial features visible
    alpha = strenght

    mask_3 = np.stack([mask, mask, mask], axis = 2)
    if preserve_lightness:
        lab_new = lab.copy()
        lab_new[:,:,1] = lab[:,:,1] * (1-mask) + (lab[:,:,1] * (1-alpha) + alpha*target_lab[1]) * mask
        lab_new[:,:,2] = lab[:,:,2] * (1-mask) + (lab[:,:,2] * (1-alpha) + alpha*target_lab[2]) * mask
        # Optionally slightly shift L to make "light/dark" target effect:
        lab_new[:,:,0] = lab[:,:,0] * (1-mask) + (lab[:,:,0] * (1-alpha*0.35) + alpha*0.35*target_lab[0]) * mask
    else:
        lab_new = lab*(1-mask_3) + (lab*(1-alpha) + alpha*target_lab.reshape(1,1,3)) * mask_3

    # Convert back to RGB/BGR
    rgb_new = color.lab2rgb(lab_new)  # returns floats 0..1
    rgb_new = np.clip(rgb_new, 0, 1)
    bgr_new = cv2.cvtColor((rgb_new*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr_new