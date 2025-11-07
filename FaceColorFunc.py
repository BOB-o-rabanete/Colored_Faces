import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh

# np.uint8 -> mem-efficient img processing and data analysis
#---------------------------------------------------------------------

#-------------#
#| FACE MASK |#
#-------------#

def get_face_mask(img: np.ndarray) -> np.ndarray:
    #later change to BiSeNet so as to properly remove eyes and lips
    """ 
    Description
        Given an image, crops a mask with only the skin and the rest as gray pixels
        eyes and lips are removed
        Uses Mediapipe Face Mesh

    ------------------
    Parameters
        img: np.ndarray
            Input image in RGB format
    ---------
    Returns
        mask: np.ndarray 
            Same size as input, non-skin areas set to [128,128,128] (gray)
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
        #left_eye = get_points(left_eye_idx)
        #right_eye = get_points(right_eye_idx)
        #lips = get_points(lips_idx)


        # ---- Create base mask ---- #
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_outline], 255)  # fill outer face area

        # Remove eyes and lips from mask #fais drastically
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

"""
For pastel or non-realistic colors, lower delta by multiplying it (e.g. 0.8 * delta) to avoid clipping.

For more realism, you can add a mild blur to the LAB channels before conversion back.

To simulate “lighter” or “darker” versions of the same color, adjust the L channel separately.
"""

def shift_skin_color(face_img: np.ndarray, color: tuple = (0, 120, 0)) -> np.ndarray :
    """ 
    Description
        Given a masked_face (skin visible, background gray), recolors the skin toward 'color'
        while preserving lightness and shade differences.
        (Improve explanation -> mention lab and vector)

    ------------------
    Parameters
        face_img : np.ndarray
            Image in RGB format, skin visible, background = [128,128,128].
        color : tuple or list of int
            Target RGB color, e.g. (0, 255, 0) for green.
    ---------
    Returns
        painted_face : np.ndarray
            Face image with shifted skin tones.
    """

    # Convert to LAB (perceptually uniform color space)
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    mask = np.any(face_img != [128, 128, 128], axis=-1)  # true for skin pixels only

    # Get mean LAB of current skin tone -> maby change to median?
    mean_lab = np.mean(lab[mask], axis=0)

    # Convert target color to LAB
    target_lab = cv2.cvtColor(
        np.uint8([[color]]), cv2.COLOR_RGB2LAB
    )[0, 0].astype(np.float32)

    # Compute shift vector
    delta = target_lab - mean_lab

    # Apply shift only to skin pixels
    shifted = lab.astype(np.float32)
    for c in range(3):
        shifted[..., c][mask] += delta[c]
    
    # Clip and convert back to uint8
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    painted_face = cv2.cvtColor(shifted, cv2.COLOR_LAB2RGB)

    return painted_face


#--------------#
#| SKIN PAINT |#
#--------------#

"""
Criar uma segunda, mais âmpla máscara

arranjar um meio termo entre as cores existentes e a nova para não haver uma tão grande discrepância
"""
def change_face_color(img: np.ndarray, color: tuple, bgr: bool = False) -> np.ndarray:
    """ 
    Description
        Given an image, returns a version of it with a painted face.

    ------------------
    Parameters
        img : np.ndarray
            Input image in RGB format.
        color : tuple or list of int
            Target RGB color, e.g. (0, 255, 0) for green.
        bgr : bool
            defines if output img is in BGR or RGB
    ---------
    Returns
        painted_face : np.ndarray
            Image with face with shifted skin tones.
    """

    # Get RGB format~
    if img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img.copy()

    # Get mask
    masked_face = get_face_mask(img_rgb)
    # Create binary mask for skin areas (everything except gray background)
    skin_mask = np.any(masked_face != [128, 128, 128], axis=-1)

    # Paint mask
    colored_face_mask = shift_skin_color(masked_face, color)

    # Get altered image
    painted_face = img_rgb.copy()
    painted_face[skin_mask] = colored_face_mask[skin_mask]
    if bgr:
        painted_face = cv2.cvtColor(painted_face, cv2.COLOR_RGB2BGR)

    return painted_face
