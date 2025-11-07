import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh

#---------------------------------------------------------------------

#-------------#
#| FACE MASK |#
#-------------#

def get_face_mask(img):
    #later change to BiSeNet so as to properly remove eyes and lips
    """ 
    Description
        Given an image, crops a mask with only the skin and the rest as gray pixels
        eyes and lips are removed
        Uses Mediapipe Face Mesh

    ------------------
    Parameters
        img: np.ndarray - input image in RGB format
    ---------
    Returns
        mask: np.ndarray - same size as input, non-skin areas set to [128,128,128] (gray)
    """
    # Initialize face mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1, 
        refine_landmarks=False,
        min_detection_confidence=0.3
    ) as face_mesh:

        results = face_mesh.process(img)
        if not results.multi_face_landmarks:
            print("🐢 ---> no face detected in image.")
            gray_bg = np.full_like(img, 128)
            return gray_bg

        # Get landmarks for the face
        h, w, _ = img.shape
        landmarks = results.multi_face_landmarks[0]


        # ---- Outer facial contour indices ---- #
        # Creating the polynomial of the detected face using some border points 
        # These points are indices from the Mediapipe FaceMesh model
        face_outline_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109, 10
        ]


        # ---- Eyes and lips indices ---- #
        # Taken from Mediapipe documentation / common face parsing setups
        left_eye_idx  = list(range(33, 133))
        right_eye_idx = list(range(362, 463))
        lips_idx      = list(range(61, 91)) + list(range(181, 211))

        # Convert normalized landmarks → pixel coords
        def get_points(idxs):
            return np.array([
                [int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)]
                for i in idxs
            ])

        face_outline = get_points(face_outline_idx)#face_outline_idx
        left_eye = get_points(left_eye_idx)
        right_eye = get_points(right_eye_idx)
        lips = get_points(lips_idx)


        # ---- Create base mask ---- #
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_outline], 255)  # fill outer face area

        # Remove eyes and lips from mask
        #cv2.fillPoly(mask, [left_eye], 0)
        #cv2.fillPoly(mask, [right_eye], 0)
        #cv2.fillPoly(mask, [lips], 0)

        # Smooth edges a bit
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        # Expand to 3 channels
        mask_3ch = cv2.merge([mask, mask, mask])

        # Make gray background
        gray_bg = np.full_like(img, 128)

        # Combine: if mask=255 → keep skin pixel; else → gray background
        mask_img = np.where(mask_3ch == 255, img, gray_bg)

        return mask_img
    
    return None

#----------------#
#| COLOUR SHIFT |#
#----------------#

def shift_skin_color(face_img, color):
    """ 
    Description
        Given a masked_face (skin visible, background gray), recolors the skin toward 'color'
    while preserving lightness and shade differences.

    ------------------
    Parameters
        face_img : np.ndarray
            Image in RGB format, skin visible, background = [128,128,128].
        color : tuple or list of int
            Target RGB color, e.g. (0, 255, 0) for green.
    ---------
    Returns
        painted_face: type - altered version of face_img with skin color as 'color'
    """
    return 

#--------------#
#| SKIN PAINT |#
#--------------#
