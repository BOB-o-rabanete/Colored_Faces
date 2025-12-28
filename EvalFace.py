"""
here are created and stored the functions to be used
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import mediapipe as mp
import dlib
dlib_detector = dlib.get_frontal_face_detector()

FR_list = ["mediapipe", "dlib_hog", "mtcnn"]



#---------------------------------------------------------------------
#-------------------#
#| FACE EVALUATION |#
#-------------------#

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(
    keep_all=True,          # detect all faces
    device=device,
    min_face_size=20,       # small faces allowed
    thresholds=[0.6, 0.7, 0.7]  # detection confidence per stage
)

def detect_mtcnn(img: np.ndarray) -> bool:
    """
    Description
        Uses MTCNN (facenet_pytorch) to detect faces.
        Returns True if at least one face is detected.

    ------------------
    Parameters
        img : np.ndarray
            Image in RGB format.

    ---------
    Returns
        : bool
            wheter or not there were detected anny fotos
        
    """

    if img.ndim == 3 and img.shape[2]==3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img.copy()

    # try to adapt style for easier recognition 
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-6)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    # Run detection 
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is None:
        return False
    
    # se quiser alterar confidence
    # valid = [p for p in probs if p is not None and p > 0.9]
    return True


def detect_dlib_hog(img: np.ndarray) -> bool:
    """
    Description
        Uses dlib HOG-based frontal face detector.
        Returns True if at least one face is detected.

    ------------------
    Parameters
        img : np.ndarray
            Image in RGB format.

    ---------
    Returns
        len(faces) > 0: bool
            wheter or not there were detected anny fotos
        
    """

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Detect faces
    faces = dlib_detector(gray, upsample_num_times=1)

    return len(faces) > 0

def detect_mediapipe(img):
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # 0: short-range (selfies, cropped faces); 1:long-range (full images)
        min_detection_confidence=0.3
    )
    result = mp_face.process(img)
    return result.detections is not None



#------------------#
#| IMG EVALUATION |#
#------------------#


def FaceDetected(img: np.ndarray, face_models: list[str] = FR_list) -> list[bool]:
    """ 
    Description
        Given an imag and list of face recognition models 
        will output whether or not eah model was able to detect a face

    ------------------
    Parameters
        img: np.ndarray
            Input image in RGB format
        face_models: list
            names of the face recognition models to evaluate the image with
    ---------
    Returns
        reults: list[bool]
            0 or 1 for each model depending on whether the model could or not detect a face
    """

    if img.ndim == 3 and img.shape[2]==3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img.copy()
        
    results = []

    for model in face_models:
        if model == "mediapipe":
            results.append(detect_mediapipe(img_rgb))
        elif model == "dlib_hog":
            results.append(detect_dlib_hog(img_rgb))
        elif model == "mtcnn":
            results.append(detect_mtcnn(img_rgb))
        else:
            raise ValueError(f"Unknown face model: {model}")

    return results
