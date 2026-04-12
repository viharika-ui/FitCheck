import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def is_visible(coords):
    return 0.0 <= coords["x"] <= 1.0 and 0.0 <= coords["y"] <= 1.0

def detect_pose(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read {image_path}")
        return None
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(imgRGB)
    
    if results.pose_landmarks is None:
        print(f"No person detected in {image_path}")
        return None
    
    landmarks = results.pose_landmarks.landmark
    keypoints = {
        "nose":           {"x": landmarks[0].x,  "y": landmarks[0].y},
        "left_shoulder":  {"x": landmarks[11].x, "y": landmarks[11].y},
        "right_shoulder": {"x": landmarks[12].x, "y": landmarks[12].y},
        "left_hip":       {"x": landmarks[23].x, "y": landmarks[23].y},
        "right_hip":      {"x": landmarks[24].x, "y": landmarks[24].y},
        "left_elbow":     {"x": landmarks[13].x, "y": landmarks[13].y},
        "right_elbow":    {"x": landmarks[14].x, "y": landmarks[14].y},
        "left_wrist":     {"x": landmarks[15].x, "y": landmarks[15].y},
        "right_wrist":    {"x": landmarks[16].x, "y": landmarks[16].y},
        "left_knee":      {"x": landmarks[25].x, "y": landmarks[25].y},
        "right_knee":     {"x": landmarks[26].x, "y": landmarks[26].y},
        "left_ankle":     {"x": landmarks[27].x, "y": landmarks[27].y},
        "right_ankle":    {"x": landmarks[28].x, "y": landmarks[28].y},
        "image_size":     {"width": img.shape[1], "height": img.shape[0]}
    }

    # Check which keypoints are actually visible in frame
    visible = {k: v for k, v in keypoints.items() 
               if k != "image_size" and is_visible(v)}
    print(f"Visible keypoints: {len(visible)}/13")
    
    mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
    
    return keypoints


if __name__ == "__main__":
    folder = "test_images"
    
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found!")
    else:
        for file in os.listdir(folder):
            if not file.endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, file)
            print(f"\nProcessing: {file}")
            result = detect_pose(path)
            print(result)