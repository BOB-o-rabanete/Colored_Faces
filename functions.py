
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
import os
import random
import re
import zipfile

#---------------------------------------------------------------------
#------------------#
#| IMG EXTRACTION |#
#------------------#

def get_subset_from_zip(zip_path: str, out_folder: str, target_age_range: tuple[int, int], target_gender: int) -> int:
    """ 
    Description
        Extracts a filtered subset of images from the UTKFace dataset ZIP file without extracting the entire archive.

    ---------
    Parameters
        zip_path: Path to the UTKFace zip file
        out_folde: Folder where the filtered images will be saved
        target_age_range:(min_age, max_age) range of ages to include
        targetgender: Gender to filter by (0 = male, 1 = female)
    ---------
    Returns
        Number of images successfully extracted
    """

    race_labels = {
        0: 'white',
        1: 'Black',
        2: 'Asian',
        3: 'Indian',
        4: 'Others'
    }

    os.makedirs(out_folder, exist_ok = True)
    pattern = re.compile(r"^(\d+)_([01])_([0-4])_")

    selected_files = []

    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            if 'utkface_aligned_cropped' not in name.lower():
                continue

            m = pattern.match(os.path.basename(name))
            if not m:
                continue

            age, gender, race = map(int, m.groups())
            if target_age_range[0] <= age <= target_age_range[1] and gender == target_gender:
                selected_files.append((name, race))

        gender_label = "male" if target_gender == 0 else "female"
        print(f"Found {len(selected_files)} images of ({target_age_range[0]} to {target_age_range[1]})-year-old {gender_label}.")

        for file, race in selected_files:
            race_folder = os.path.join(out_folder, race_labels.get(race, 'unknown')) #seperate imgs by race
            os.makedirs(race_folder, exist_ok=True)

            filename = os.path.basename(file)
            out_path = os.path.join(race_folder, filename)

            with zf.open(file) as source, open(out_path, 'wb') as target:
                target.write(source.read())
    

    print(f" Successfully extracted {len(selected_files)} images to '{out_folder}'")
    return len(selected_files)



def image_quality_score(img_path: str) -> float:
    """ 
    Description
        Compute an image quality score based on the Laplacian variance method(function estimates how sharp or blurry an image is).

    ---------
    Parameters
        img_path: The file path to the image
    ---------
    Returns
        The Laplacian variance score
        Returns 0.0 if the image cannot be read
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    laplac_var = cv2.Laplacian(img, cv2.CV_64F).var() #Laplacio used for blurinnes
    return laplac_var

def abv_avg_quality_randomizer(folder_path: str, n_imgs:int = 10) -> dict[str: list]:
    """ 
    Description
        Per race select a random sample of above-average quality images with size n_imgs

    ---------
    Parameters
        folder_path: The root directory containing subfolders of images
        n_imgs: Optional
                The number of images to randomly sample per subfolder (default is 10)
    ---------
    Returns
        A dictionary where keys are subfolder names and values are lists of selected image paths.

    """
    sampeld = {}

    for race in os.listdir(folder_path):
        race_path = os.path.join(folder_path, race) #till like 'white'
        img_paths = [os.path.join(race_path, f) for f in os.listdir(race_path)] #get path of each img

        scores = {img_path: image_quality_score(img_path) for img_path in img_paths} 
        avg = np.mean(scores.values())
        good_imgs = [img for img, scor in scores.items() if scor >= avg]
        if len(good_imgs)<n_imgs:
            print(f' Warning: number of existing good images is infirior to expected limit {n_imgs} for race {race}')
        sampeld[race] = random.sample(good_imgs, min(n_imgs, len(good_imgs)))
    
    return sampeld

#---------------------------------------------------------------------

#--------------#
#| FACE COLOR |#
#--------------#
def rgb_to_lab(rgb):
    """ 
    Description
    ---------
    Parameters
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
    Parameters
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
    Parameters
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
    Parameters
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
    Parameters
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
    Parameters
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