
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

