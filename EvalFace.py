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

# ----------------------------
# MediaPipe
# ----------------------------
_mp_face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.3
)

# ----------------------------
# dlib HOG
# ----------------------------
_dlib_detector = dlib.get_frontal_face_detector()

# ----------------------------
# MTCNN
# ----------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_mtcnn_detector = MTCNN(
    keep_all=True,
    device=_device,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.5]
)


def detect_mediapipe(img_rgb: np.ndarray) -> bool:
    results = _mp_face_detector.process(img_rgb)
    return results.detections is not None


def detect_dlib_hog(img_rgb: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _dlib_detector(gray, 1)
    return len(faces) > 0


def detect_mtcnn(img_rgb: np.ndarray) -> bool:
    boxes, _ = _mtcnn_detector.detect(img_rgb)
    return boxes is not None and len(boxes) > 0


#------------------#
#| IMG EVALUATION |#
#------------------#


def FaceDetected(img_bgr: np.ndarray, face_models: list[str] = FR_list) -> list[bool]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = []

    for model in face_models:
        if model == "mediapipe":
            results.append(detect_mediapipe(img_rgb) or detect_mediapipe(img_bgr))

        elif model == "dlib_hog":
            results.append(detect_dlib_hog(img_rgb) or detect_dlib_hog(img_bgr))

        elif model == "mtcnn":
            results.append(detect_mtcnn(img_rgb) or detect_mtcnn(img_bgr))

        else:
            raise ValueError(f"Unknown face model: {model}")

    return results


#-----------#
#| CLEANUP |#
#-----------#

def close_face_detectors():
    _mp_face_detector.close()
