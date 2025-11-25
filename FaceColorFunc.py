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

    return painted_face, delta


def smooth_borderline_og(new_img: np.ndarray, mask_img: np.ndarray, blur_radius: int = 35) -> np.ndarray:
    """
    Description
        Diffuses the color of the recolored skin slightly into the nearby background
        to create a smoother, more natural transition without bright halo artifacts.

    ------------------
    Parameters
        new_img : np.ndarray
            Image with recolored face (RGB format).
        mask_img : np.ndarray
            Masked image used to produce recoloring (background = [128,128,128]).
        blur_radius : int
            Radius (in pixels) over which color diffusion should occur.
    ---------
    Returns
        result : np.ndarray
            Smoothed image where the border between face and background
            gradually transitions with mixed colors.
    """

    # --- Prepare data - for blending need to be float [0,1]
    img = new_img.astype(np.float32)
    if len(mask_img.shape) == 2:  # grayscale mask
        mask_gray = (mask_img != 128).astype(np.uint8)
    else:
        mask_gray = np.any(mask_img != [128, 128, 128], axis=-1).astype(np.uint8)


    # --- Compute distance from non-face regions to the face contour
    dist_to_face = cv2.distanceTransform(1 - mask_gray, cv2.DIST_L2, 5)
    dist_to_face = np.clip(dist_to_face, 0, blur_radius)
    dist_norm = 1 - (dist_to_face / blur_radius)  # 1 near face, 0 far away

    # --- Compute average skin color
    skin_pixels = img[mask_gray.astype(bool)]
    if len(skin_pixels) == 0:
        return new_img  # fallback if no skin detected
    mean_skin_color = np.mean(skin_pixels, axis=0)

    # --- Blend pixels near the border toward the mean skin color
    fade_mask = np.expand_dims(dist_norm, axis=-1)
    result = img.copy()
    result = result * (1 - fade_mask) + mean_skin_color * fade_mask

    # --- Keep original skin region intact
    result[mask_gray.astype(bool)] = img[mask_gray.astype(bool)]

    return np.clip(result, 0, 255).astype(np.uint8)  


def smooth_borderline(new_img: np.ndarray,
                      mask_img: np.ndarray,
                      delta: np.ndarray,
                      target_color: tuple,
                      blur_radius: int = 40,
                      mode: str = "delta") -> np.ndarray:
    """
    Diffuses skin color effect outward from the face into surrounding pixels.

    ------------------
    Parameters
        new_img      : recolored face (RGB)
        mask_img     : mask used during recoloring (non-face = [128,128,128])
        delta        : RGB vector applied to recolor skin (target - base)
        target_color : target color (RGB), used in mode="blend"
        blur_radius  : max distance (px) for color diffusion
        mode         : "delta" (default) or "blend"
                       - "delta":    background += delta * weight
                       - "blend":    background = bg*(1-w) + target_color*w

    Returns
        result_img : final smoothed image
    """

    # Work in float
    img = new_img.astype(np.float32)

    # Convert mask into binary (face = 1, background = 0)
    if len(mask_img.shape) == 2:        # grayscale mask
        mask_gray = (mask_img != 128).astype(np.uint8)
    else:
        mask_gray = np.any(mask_img != [128,128,128], axis=-1).astype(np.uint8)

    # Distance from background pixel to nearest face pixel
    dist = cv2.distanceTransform(1 - mask_gray, cv2.DIST_L2, 5)
    dist = np.clip(dist, 0, blur_radius)
    weight = 1.0 - (dist / blur_radius)   # 1 near face, 0 far away

    weight_3 = np.expand_dims(weight, axis=-1)

    # --- A: delta-based diffusion (default)
    if mode == "delta":
        result = img.copy()
        result += delta * weight_3
        # Keep skin unchanged
        result[mask_gray.astype(bool)] = img[mask_gray.astype(bool)]
        return np.clip(result, 0, 255).astype(np.uint8)

    # --- B: blend-to-target-color diffusion
    elif mode == "blend":
        target = np.array(target_color, dtype=np.float32)
        result = img * (1 - weight_3) + target * weight_3
        # Preserve actual recolored skin region
        result[mask_gray.astype(bool)] = img[mask_gray.astype(bool)]
        return np.clip(result, 0, 255).astype(np.uint8)

    else:
        raise ValueError("mode must be 'delta' or 'blend'")

'''
    will create a zone arround the face area determined by face_mask
    og_pixel, pixel chosen from anywhere that has not been colores(outside the mask)
    the new delta will be given by the vector calculated from the og pixel to the n1_pixel (product of delta on og_pixel)
    all the pixels on the blur_raduois will be shifted according tho the new delta
    a new_mask will be created simmilar to the other but containing the new_shifted pixels
    
    returns changed_img, new_delta, new_mask
'''

def smooth_color_transition(
        new_img: np.ndarray,
        face_mask: np.ndarray,
        delta: np.ndarray,
        blur_radius: int = 10,
        use_constant_decay: bool = False,
    ):
    
    """
    Performs ONE outward expansion:
        - find border ring around face mask
        - shift its pixels using delta
        - expand mask
        - attenuate delta (B or C)

    Inputs:
        new_img      : RGB float32 or uint8 image
        face_mask    : binary mask (0 = bg, 1 = face)
        delta        : (3,) shift vector
        blur_radius  : thickness of expansion
        use_constant_decay : if True → method C, else B

    Returns:
        updated_img
        new_delta (3,)
        new_mask (binary)   
    """

    # Ensure correct formats
    img = new_img.astype(np.float32)
    mask = face_mask.astype(np.uint8)

    # === 1. Distance to nearest face pixel ===
    # mask should be 8-bit 0/1 for distanceTransform
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)

    # Ring = within blur_radius but NOT inside mask
    ring = np.logical_and(dist > 0, dist <= blur_radius)

    if not np.any(ring):
        # No more pixels to fill
        return img.astype(np.uint8), delta, mask

    # === 2. Apply delta to ring pixels ===
    ring_pixels = img[ring]                 # shape (N, 3)
    shifted = ring_pixels + delta           # delta must be (3,) !!
    shifted = np.clip(shifted, 0, 255)

    updated_img = img.copy()
    updated_img[ring] = shifted

    # === 3. Decay delta ===
    if use_constant_decay:
        # Method C — linear decay to zero
        decay = max(0.0, 1.0 - (1.0 / blur_radius))
    else:
        # Method B — exponential radial decay
        # larger k → slower fade
        k = 8.0
        decay = np.exp(-1.0 / k)

    new_delta = delta * decay  # stays shape (3,)

    # === 4. Expand mask with ring
    new_mask = mask.copy()
    new_mask[ring] = 1

    return updated_img.astype(np.uint8), new_delta, new_mask

def diffuse_color_recursive(base_img: np.ndarray,
                            face_mask: np.ndarray,
                            delta: np.ndarray,
                            blur_radius: int = 3,
                            use_constant_decay: bool = False,
                            stop_norm: float = 0.4,
                            max_iters: int = 2):
    """
    Repeatedly smooths color outward from the face mask until
    delta magnitude becomes tiny or max_iters reached.

    Inputs:
        base_img          : RGB image
        face_mask         : binary mask
        delta             : (3,) vector
        blur_radius       : ring thickness
        use_constant_decay: method C if True, else B
        stop_norm         : stop if ||delta|| < stop_norm
        max_iters         : safety limit

    Returns:
        smoothed_img (uint8 RGB)
    """

    current_img = base_img.copy()
    #current_mask = face_mask.astype(np.uint8)
    face_mask = face_mask.astype(np.uint8)
    if len(face_mask.shape) == 2:  # grayscale mask
        current_mask = (face_mask != 128).astype(np.uint8)
    else:
        current_mask = np.any(face_mask != [128, 128, 128], axis=-1).astype(np.uint8)
    #print(type(current_mask), current_mask)
    current_delta = delta.astype(np.float32)

    for _ in range(max_iters):

        # stop if delta is nearly zero
        if np.linalg.norm(current_delta) < stop_norm:
            break

        current_img, current_delta, current_mask = smooth_color_transition(
            current_img,
            current_mask,
            current_delta,
            blur_radius=blur_radius,
            use_constant_decay=use_constant_decay
        )
        print('n ----------------------- n')

    return current_img


def apply_translucent_falloff_tint(img_rgb, face_mask, target_color, blur_radius=25):
    """
    Softly tints ONLY the background pixels toward the target_color.
    Tint strength fades with distance from the face edge (face contours stay intact).
    """

    # Prepare
    img = img_rgb.astype(np.float32)
    target = np.array(target_color, dtype=np.float32)

    # Normalize mask into binary 0/1
    if len(face_mask.shape) == 2:
        mask = (face_mask != 128).astype(np.uint8)
    else:
        mask = np.any(face_mask != [128,128,128], axis=-1).astype(np.uint8)

    # Calculate distance from face to background
    # distanceTransform works on 8-bit single-channel
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)

    # Normalize distances into α range
    dist = np.clip(dist, 0, blur_radius)
    alpha = 1 - (dist / blur_radius)   # 1 near boundary, 0 far away
    
    # --- KEY FIX: No tint inside the face ---
    alpha[mask == 1] = 0.0

    alpha = np.expand_dims(alpha, axis=-1)

    # Only tint background
    tinted = img * (1 - alpha) + target * alpha

    # Restore face pixels from the original input image
    tinted[mask == 1] = img[mask == 1]

    return np.clip(tinted, 0, 255).astype(np.uint8)


def delta_falloff_recolor(img_rgb, face_mask, delta, blur_radius=40):
    """
    Smoothly extends LAB-delta recoloring outside the face,
    fading with distance from the face boundary.

    img_rgb: original full RGB image
    face_mask: binary (0/1) face mask
    delta: 3-element LAB shift vector (from shift_skin_color)
    """

    # Convert to LAB (float for math)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Ensure binary mask
    if len(face_mask.shape) == 3:
        mask = np.any(face_mask != [128,128,128], axis=-1).astype(np.uint8)
    else:
        mask = (face_mask != 128).astype(np.uint8)

    # Distance from face border
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)

    # Normalize distances
    dist = np.clip(dist, 0, blur_radius)
    fade = 1 - (dist / blur_radius)     # 1 at border, 0 far away

    fade = np.expand_dims(fade, axis=-1)   # shape (H,W,1)

    # Apply LAB delta only to background
    # At border: full delta, far away: near 0 delta
    for c in range(3):
        lab[..., c] += delta[c] * fade[..., 0] * (1 - mask)  # mask==0 outside

    # Convert back to RGB
    recolored = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),
                             cv2.COLOR_LAB2RGB)

    return recolored


#--------------#
#| SKIN PAINT |#
#--------------#

"""
Criar uma segunda, mais âmpla máscara

arranjar um meio termo entre as cores existentes e a nova para não haver uma tão grande discrepância
"""
def change_face_color(img: np.ndarray, color: tuple, bgr: bool = False, smooth: float = -1.0) -> np.ndarray:
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
        smooth : float (optional)
            represents the blur radius, when positive creates a smoother merge between the colored face and the background
    ---------
    Returns
        painted_face : np.ndarray
            Image with face with shifted skin tones.
    """

    # Get RGB format    
    if img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img.copy()

    # Get mask
    masked_face = get_face_mask(img_rgb) 
    # Create binary mask for skin areas (everything except gray background)
    skin_mask = np.any(masked_face != [128, 128, 128], axis=-1)

    # Paint mask
    colored_face_mask, delta = shift_skin_color(masked_face, color)

    # Get altered image / Smoothing -> optional
    """
    if smooth>0:
        smooth = (smooth//2)*2 + 1 #must be odd
        painted_face = smooth_borderline(colored_face_mask, skin_mask, blur_radius=smooth)
    else:
        painted_face = img_rgb.copy()
        painted_face[skin_mask] = colored_face_mask[skin_mask]

    if bgr:
        painted_face = cv2.cvtColor(painted_face, cv2.COLOR_RGB2BGR)
    """
    painted_face = img_rgb.copy()
    painted_face[skin_mask] = colored_face_mask[skin_mask]

    if bgr:
        painted_face = cv2.cvtColor(painted_face, cv2.COLOR_RGB2BGR)

    if smooth>0:
        smooth = (smooth//2)*2 + 1 #must be odd
        #painted_face = diffuse_color_recursive(painted_face, masked_face, delta= delta)
        painted_face = delta_falloff_recolor(painted_face, masked_face, delta, blur_radius=40)
        print(":(")
        #painted_face2 = smooth_borderline_v2(painted_face, masked_face, blur_radius=smooth)
        #return painted_face2
    
    return painted_face
