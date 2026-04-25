import os
import io
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from rembg import remove


def load_image(path, size=None):
    img = Image.open(path).convert("RGBA")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA))

def apply_mask_to_garment(garment_img, mask):
    garment_np = np.array(garment_img)
    if mask.shape[:2] != garment_np.shape[:2]:
        mask = cv2.resize(mask, (garment_np.shape[1], garment_np.shape[0]))
    garment_np[:, :, 3] = mask
    return Image.fromarray(garment_np)

def resize_garment_to_body(garment_img, body_keypoints):
    return garment_img

def get_garment_position(body_keypoints, garment_size):
    return (0, 0)


def _remove_garment_bg(shirt_cv):
    """Remove garment background. Falls back to GrabCut for dark shirts."""
    # Try rembg first
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(shirt_cv, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
    result_pil = Image.open(io.BytesIO(remove(buf.getvalue()))).convert('RGBA')
    shirt_bgra = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGBA2BGRA)
    shirt_bgr  = shirt_bgra[:, :, :3]
    alpha      = shirt_bgra[:, :, 3].copy()

    # If rembg gave empty/poor alpha → fallback to GrabCut
    if alpha.max() < 10:
        print("[overlay] rembg failed, using GrabCut fallback...")
        h, w = shirt_cv.shape[:2]
        mask_gc = np.zeros((h, w), np.uint8)
        rect = (int(w*0.05), int(h*0.05), int(w*0.90), int(h*0.90))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(shirt_cv, mask_gc, rect, bgd, fgd, 10, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
        shirt_bgr = shirt_cv.copy()

    # Clean alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    return shirt_bgr, alpha


def overlay_garment(
    body_img_path,
    garment_img_path,
    body_keypoints,
    garment_mask=None,
    use_seamless=False,
    output_path="output_overlay.png"
):
    # ── Load images ──
    body_cv  = cv2.imread(body_img_path)
    shirt_cv = cv2.imread(garment_img_path)

    if body_cv is None:
        raise FileNotFoundError(f"Cannot read body image: {body_img_path}")
    if shirt_cv is None:
        raise FileNotFoundError(f"Cannot read garment image: {garment_img_path}")

    ph, pw = body_cv.shape[:2]

    # ── Remove garment background ──
    shirt_bgr, alpha = _remove_garment_bg(shirt_cv)

    # ── Landmarks ──
    ls = np.float32(body_keypoints['left_shoulder'])
    rs = np.float32(body_keypoints['right_shoulder'])
    lh = np.float32(body_keypoints['left_hip'])
    rh = np.float32(body_keypoints['right_hip'])

    if ls[0] > rs[0]: ls, rs = rs, ls
    if lh[0] > rh[0]: lh, rh = rh, lh

    sw_body = rs[0] - ls[0]
    cy_sho  = (ls[1] + rs[1]) / 2
    cy_hip  = (lh[1] + rh[1]) / 2
    height  = cy_hip - cy_sho
    cx      = (ls[0] + rs[0]) / 2
    half_w  = sw_body * 0.80
    top_y   = cy_sho - height * 0.18
    bot_y   = cy_hip + height * 0.05

    # ── Shirt bounding box ──
    coords = cv2.findNonZero((alpha > 30).astype(np.uint8))
    if coords is None:
        print("[overlay] Empty shirt mask, returning original")
        return Image.fromarray(cv2.cvtColor(body_cv, cv2.COLOR_BGR2RGB))
    x, y, w, h = cv2.boundingRect(coords)

    # ── Homography ──
    src = np.float32([[x,     y    ], [x+w,   y    ],
                      [x+w,   y+h  ], [x,     y+h  ]])
    dst = np.float32([[cx-half_w, top_y], [cx+half_w, top_y],
                      [cx+half_w, bot_y], [cx-half_w, bot_y]])

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        print("[overlay] Homography failed")
        return Image.fromarray(cv2.cvtColor(body_cv, cv2.COLOR_BGR2RGB))

    warped_shirt = cv2.warpPerspective(shirt_bgr, H, (pw, ph))
    warped_alpha = cv2.warpPerspective(alpha,     H, (pw, ph))

    # ── Blend ──
    a  = warped_alpha.astype(np.float32) / 255.0
    a3 = cv2.merge([a, a, a])
    result = (warped_shirt.astype(np.float32) * a3 +
              body_cv.astype(np.float32) * (1 - a3)).astype(np.uint8)

    # ── Clip to body mask ──
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        out = seg.process(cv2.cvtColor(body_cv, cv2.COLOR_BGR2RGB))
        body_mask = (out.segmentation_mask > 0.5).astype(np.uint8) * 255

    mask3  = cv2.merge([body_mask, body_mask, body_mask])
    result = np.where(mask3 > 127, result, body_cv)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"[overlay] Saved → {output_path}")
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
