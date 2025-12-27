"""
here are created and stored the functions to be used
"""

import cv2
import numpy as np
# ---- DLIB ----
import dlib
dlib_hog = dlib.get_frontal_face_detector()
# ---- MTCNN ----
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True, device='cpu')
# ---- MediaPipe ----
import mediapipe as mp
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)
# ---- RetinaFace ----
from retina_face import RetinaFace


#---------------------------------------------------------------------
#------------------#
#| IMG EVALUATION |#
#------------------#

def FaceDetected(img: np.ndarray, face_models: list[str]) -> list[bool]:
    #later change to BiSeNet so as to properly remove eyes and lips
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

    return