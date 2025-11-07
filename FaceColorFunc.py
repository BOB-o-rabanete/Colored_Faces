import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh

#---------------------------------------------------------------------

#-------------#
#| FACE MASK |#
#-------------#

def get_face_mask(img):
    """ 
    Description
        Given an image, crops a mask with only the skin and the rest as white pixels
        Uses Mediapipe Face Mesh

    ------------------
    Parameters
        img: np.ndarray - image of a face
    ---------
    Returns
        mask: np.ndarray - a copy of the img but, anywhere that was not recognized as skin is defined as [0,0,0], white
    """
    # Initialize face mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(img)
        if not results.multi_face_landmarks:
            print("⚠️ No face detected in image.")
            return np.zeros_like(img)

        # Get landmarks for the first face
        h, w, _ = img.shape
        landmarks = results.multi_face_landmarks[0]

        # Define polygon points for the facial skin region
        # These are indices from the Mediapipe FaceMesh model
        # representing outer face contour and cheeks
        face_outline_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109, 10
        ]

        # Convert normalized landmark coords to pixel coords
        points = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for i, lm in enumerate(landmarks.landmark)
            if i in face_outline_idx
        ])

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Optionally smooth the mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Apply mask to original image
        mask_3ch = cv2.merge([mask, mask, mask])
        mask_img = cv2.bitwise_and(img, mask_3ch)

        return mask_img

    return

#----------------#
#| COLOUR SHIFT |#
#----------------#

#--------------#
#| SKIN PAINT |#
#--------------#
