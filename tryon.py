"""
tryon.py  —  Virtual Clothing Try-On Pipeline
=============================================
Pipeline:
  1. Pose estimation (MediaPipe)
  2. Body segmentation (U2Net / rembg)
  3. Garment mask extraction
  4. Garment warping & overlay
  5. Save final result

Usage:
  python tryon.py --body person.jpg --garment shirt.png --output result.jpg
"""

import cv2
import numpy as np
import argparse
import os
from PIL import Image

import mediapipe as mp

# Import overlay utilities
from modules.overlay import (
    overlay_garment,
    apply_mask_to_garment,
    resize_garment_to_body,
    load_image,
)

# ── Constants ────────────────────────────────────────────────────────────────
POSE_LANDMARK = mp.solutions.pose.PoseLandmark
DEFAULT_OUTPUT = "tryon_result.jpg"


# ── 1. Pose Estimation ───────────────────────────────────────────────────────
def extract_keypoints(image_path):
    """
    Run MediaPipe Pose on body image.
    Returns dict of landmark name -> (pixel_x, pixel_y)
    """
    mp_pose = mp.solutions.pose
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Body image not found: {image_path}")

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(img_rgb)

    if not results.pose_landmarks:
        raise RuntimeError("No pose landmarks detected. Check body image quality.")

    landmarks = results.pose_landmarks.landmark

    def px(landmark):
        return (int(landmark.x * w), int(landmark.y * h))

    keypoints = {
        'left_shoulder':  px(landmarks[POSE_LANDMARK.LEFT_SHOULDER]),
        'right_shoulder': px(landmarks[POSE_LANDMARK.RIGHT_SHOULDER]),
        'left_elbow':     px(landmarks[POSE_LANDMARK.LEFT_ELBOW]),
        'right_elbow':    px(landmarks[POSE_LANDMARK.RIGHT_ELBOW]),
        'left_wrist':     px(landmarks[POSE_LANDMARK.LEFT_WRIST]),
        'right_wrist':    px(landmarks[POSE_LANDMARK.RIGHT_WRIST]),
        'left_hip':       px(landmarks[POSE_LANDMARK.LEFT_HIP]),
        'right_hip':      px(landmarks[POSE_LANDMARK.RIGHT_HIP]),
        'nose':           px(landmarks[POSE_LANDMARK.NOSE]),
    }

    print("[pose] Keypoints extracted:")
    for k, v in keypoints.items():
        print(f"       {k:20s}: {v}")

    return keypoints


# ── 2. Body Segmentation ─────────────────────────────────────────────────────
def segment_body(image_path):
    """
    Segment body from background using rembg (U2Net under the hood).
    Returns:
        body_rgba : PIL RGBA image with background removed
        mask      : numpy (H, W) uint8 array, 255 = person
    """
    try:
        from rembg import remove
        print("[segmentation] Using rembg (U2Net)...")
        with open(image_path, "rb") as f:
            input_data = f.read()
        output_data = remove(input_data)
        body_rgba = Image.open(__import__('io').BytesIO(output_data)).convert("RGBA")
        mask = np.array(body_rgba)[:, :, 3]
        print("[segmentation] Body segmentation complete.")
        return body_rgba, mask
    except ImportError:
        print("[segmentation] rembg not installed. Falling back to GrabCut.")
        return segment_body_grabcut(image_path)


def segment_body_grabcut(image_path):
    """
    Fallback: GrabCut-based body segmentation using OpenCV.
    """
    img_bgr = cv2.imread(image_path)
    h, w = img_bgr.shape[:2]

    mask_gc = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.9))

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask_gc, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    img_rgba[:, :, 3] = fg_mask
    body_pil = Image.fromarray(img_rgba)
    print("[segmentation] GrabCut segmentation complete.")
    return body_pil, fg_mask


# ── 3. Garment Mask Extraction ───────────────────────────────────────────────
def extract_garment_mask(garment_path):
    """
    Remove background from garment image using rembg or simple thresholding.
    Returns:
        garment_rgba : PIL RGBA
        mask         : numpy (H, W) uint8
    """
    try:
        from rembg import remove
        with open(garment_path, "rb") as f:
            data = f.read()
        out = remove(data)
        garment_rgba = Image.open(__import__('io').BytesIO(out)).convert("RGBA")
        mask = np.array(garment_rgba)[:, :, 3]
        print("[garment] Garment mask extracted via rembg.")
        return garment_rgba, mask
    except ImportError:
        print("[garment] rembg not available. Using alpha channel as mask.")
        garment_rgba = Image.open(garment_path).convert("RGBA")
        mask = np.array(garment_rgba)[:, :, 3]
        return garment_rgba, mask


# ── 4. Compute Body Measurements from Keypoints ──────────────────────────────
def compute_body_measurements(keypoints):
    """
    Return useful distances derived from pose keypoints.
    """
    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    measurements = {}

    if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
        measurements['shoulder_width'] = dist(
            keypoints['left_shoulder'], keypoints['right_shoulder']
        )

    if 'left_shoulder' in keypoints and 'left_hip' in keypoints:
        measurements['torso_height'] = dist(
            keypoints['left_shoulder'], keypoints['left_hip']
        )

    if 'left_hip' in keypoints and 'right_hip' in keypoints:
        measurements['hip_width'] = dist(
            keypoints['left_hip'], keypoints['right_hip']
        )

    print("[measurements]", {k: f"{v:.1f}px" for k, v in measurements.items()})
    return measurements


# ── 5. Visualise Keypoints (debug helper) ────────────────────────────────────
def draw_keypoints(image_path, keypoints, save_path="debug_keypoints.jpg"):
    img = cv2.imread(image_path)
    for name, (x, y) in keypoints.items():
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(img, name[:5], (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.imwrite(save_path, img)
    print(f"[debug] Keypoint image saved to {save_path}")


# ── 6. Main Try-On Pipeline ──────────────────────────────────────────────────
def run_tryon(body_path, garment_path, output_path=DEFAULT_OUTPUT,
              use_seamless=False, debug=False):
    """
    Full virtual try-on pipeline.

    Parameters
    ----------
    body_path     : str  - path to person/body image
    garment_path  : str  - path to clothing/garment image
    output_path   : str  - where to save the final result
    use_seamless  : bool - use Poisson seamless blending (better quality, slower)
    debug         : bool - save intermediate debug images
    """
    print("\n" + "=" * 55)
    print("   Virtual Try-On Pipeline Starting")
    print("=" * 55)

    # ── Step 1: Pose Estimation ──────────────────────────────
    print("\n[Step 1/5] Pose Estimation...")
    keypoints = extract_keypoints(body_path)

    if debug:
        draw_keypoints(body_path, keypoints, "debug_keypoints.jpg")

    # ── Step 2: Body Segmentation ────────────────────────────
    print("\n[Step 2/5] Body Segmentation...")
    body_rgba, body_mask = segment_body(body_path)

    if debug:
        body_rgba.save("debug_body_segmented.png")
        print("[debug] Body segmentation saved.")

    # ── Step 3: Garment Mask ─────────────────────────────────
    print("\n[Step 3/5] Garment Mask Extraction...")
    garment_rgba, garment_mask = extract_garment_mask(garment_path)

    if debug:
        garment_rgba.save("debug_garment_masked.png")
        print("[debug] Garment mask saved.")

    # ── Step 4: Body Measurements ────────────────────────────
    print("\n[Step 4/5] Computing Body Measurements...")
    measurements = compute_body_measurements(keypoints)

    # ── Step 5: Overlay & Blend ──────────────────────────────
    print("\n[Step 5/5] Overlaying Garment onto Body...")
    result = overlay_garment(
        body_img_path=body_path,
        garment_img_path=garment_path,
        body_keypoints=keypoints,
        garment_mask=garment_mask,
        use_seamless=use_seamless,
        output_path=output_path
    )

    print("\n" + "=" * 55)
    print(f"   Try-On Complete! Result saved to: {output_path}")
    print("=" * 55 + "\n")
    return result


# ── CLI Entry Point ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Virtual Clothing Try-On System"
    )
    parser.add_argument("--body",     required=True,  help="Path to body/person image")
    parser.add_argument("--garment",  required=True,  help="Path to garment image")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--seamless", action="store_true",
                        help="Use Poisson seamless blending (better quality)")
    parser.add_argument("--debug",    action="store_true",
                        help="Save intermediate debug images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.body):
        print(f"[ERROR] Body image not found: {args.body}")
        exit(1)
    if not os.path.exists(args.garment):
        print(f"[ERROR] Garment image not found: {args.garment}")
        exit(1)

    run_tryon(
        body_path=args.body,
        garment_path=args.garment,
        output_path=args.output,
        use_seamless=args.seamless,
        debug=args.debug
    )
