import cv2
import dlib
import numpy as np
import os
import math
import time
from sklearn.neighbors import NearestNeighbors

# Constants expected by the rest of the codebase. You can customize or import from main module.
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
DIST_COEFFS = np.zeros((4, 1))

# load dlib models lazily
_detector = dlib.get_frontal_face_detector()
_predictor = None
_face_recognizer = None


def _ensure_models():
    global _predictor, _face_recognizer
    if _predictor is None:
        if not os.path.exists(SHAPE_PREDICTOR_PATH):
            raise FileNotFoundError(f"shape predictor not found at {SHAPE_PREDICTOR_PATH}")
        _predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    if _face_recognizer is None:
        if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
            raise FileNotFoundError(f"face recognition model not found at {FACE_RECOGNITION_MODEL_PATH}")
        _face_recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)


def estimate_pose(landmarks, img_w, img_h, camera_matrix=None):
    """Estimate yaw/pitch/roll and return (yaw_deg, pitch_deg, roll_deg, rvec, tvec).
    Compatible with the original implementation.
    """
    # define simple camera matrix if not provided
    if camera_matrix is None:
        CAMERA_MATRIX = np.array([
            [img_w, 0, img_w / 2.0],
            [0, img_w, img_h / 2.0],
            [0, 0, 1.0]
        ], dtype="double")
    else:
        CAMERA_MATRIX = camera_matrix

    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (30)
        (0.0, -330.0, -65.0),        # Chin (8)
        (-225.0, 170.0, -135.0),     # Left eye left corner (36)
        (225.0, 170.0, -135.0),      # Right eye right corner (45)
        (-150.0, -150.0, -125.0),    # Left mouth corner (48)
        (150.0, -150.0, -125.0)      # Right mouth corner (54)
    ], dtype="double")

    image_points_selected = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")

    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points_selected, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0, None, None

    rot_matrix, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rot_matrix, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]

    yaw_deg = -float(yaw[0])
    pitch_deg = float(pitch[0])
    roll_deg = -float(roll[0])

    return yaw_deg, pitch_deg, roll_deg, rvec, tvec


def get_face_encoding(image_rgb, face_rect):
    """Compute face encoding using dlib model for a given RGB image and dlib rectangle."""
    _ensure_models()
    landmarks = _predictor(image_rgb, face_rect)
    return np.array(_face_recognizer.compute_face_descriptor(image_rgb, landmarks))


def load_dataset_encodings(dataset_dir="dataset"):
    known_encodings = []
    known_names = []
    known_poses = {}

    if not os.path.exists(dataset_dir):
        return np.array([]), [], {}

    _ensure_models()

    for user_name in os.listdir(dataset_dir):
        user_path = os.path.join(dataset_dir, user_name)
        if not os.path.isdir(user_path):
            continue
        current_user_encodings = []
        current_user_poses_data = {"frontale": [], "gauche": [], "droite": [], "haut": [], "bas": []}
        for img_name in os.listdir(user_path):
            if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                continue
            img_path = os.path.join(user_path, img_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = _detector(gray, 1)
            if len(faces) != 1:
                continue
            face = faces[0]
            try:
                encoding = get_face_encoding(img_rgb, face)
                current_user_encodings.append(encoding)
            except Exception:
                continue
            try:
                landmarks = _predictor(img_rgb, face)
                h, w = img_rgb.shape[:2]
                yaw, pitch, roll, _, _ = estimate_pose(landmarks, w, h)
            except Exception:
                continue
            for pose_key in current_user_poses_data.keys():
                if pose_key in img_name.lower():
                    current_user_poses_data[pose_key].append((yaw, pitch, roll))
                    break
        if current_user_encodings:
            avg_encoding = np.mean(current_user_encodings, axis=0)
            known_encodings.append(avg_encoding)
            known_names.append(user_name)
            avg_poses_for_user = {}
            for pose_key, lst in current_user_poses_data.items():
                if lst:
                    avg_poses_for_user[pose_key] = np.mean(lst, axis=0)
            known_poses[user_name] = avg_poses_for_user
    return np.array(known_encodings), known_names, known_poses


def capture_images_headless(user_name, num_images_per_pose=40, output_dir="dataset", camera_index=0, progress_callback=None):
    """Capture images without GUI. progress_callback(status: str, info: dict) optional."""
    user_output_dir = os.path.join(output_dir, user_name)
    os.makedirs(user_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    poses = ["frontale", "gauche", "droite", "haut", "bas"]
    MIN_TIME_BETWEEN_SAVES = 0.7
    STABLE_FRAMES_REQUIRED = 3
    STABILITY_THRESH = 20

    captured_count = 0
    for pose in poses:
        if progress_callback:
            progress_callback('pose_start', {'pose': pose})
        time.sleep(1.0)
        current_pose_count = 0
        last_save_time = 0
        last_centers = []
        while current_pose_count < num_images_per_pose:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _detector(gray)
            if len(faces) == 1:
                fr = faces[0]
                x, y, w, h = fr.left(), fr.top(), fr.width(), fr.height()
                cx = x + w / 2
                cy = y + h / 2
                last_centers.append((cx, cy))
                if len(last_centers) > STABLE_FRAMES_REQUIRED:
                    last_centers.pop(0)
                stable = False
                if len(last_centers) == STABLE_FRAMES_REQUIRED:
                    max_dist = 0
                    for i in range(len(last_centers)):
                        for j in range(i+1, len(last_centers)):
                            dx = last_centers[i][0] - last_centers[j][0]
                            dy = last_centers[i][1] - last_centers[j][1]
                            dist = math.hypot(dx, dy)
                            if dist > max_dist:
                                max_dist = dist
                    if max_dist <= STABILITY_THRESH:
                        stable = True
                now = time.time()
                if stable and (now - last_save_time >= MIN_TIME_BETWEEN_SAVES):
                    img_path = os.path.join(user_output_dir, f"{user_name}_{pose}_{current_pose_count:03d}.jpg")
                    cv2.imwrite(img_path, frame)
                    current_pose_count += 1
                    captured_count += 1
                    last_save_time = now
                    if progress_callback:
                        progress_callback('image_saved', {'pose': pose, 'count': current_pose_count, 'path': img_path})
            else:
                last_centers = []
        if progress_callback:
            progress_callback('pose_done', {'pose': pose})
        time.sleep(1.0)
    cap.release()
    return {'user': user_name, 'total_captured': captured_count}


def authenticate_sequence(known_encodings, known_names, known_poses, camera_index=0, total_challenges=2, challenge_timeout=12, progress_callback=None, status_callback=None, frame_callback=None, debug_poses=False):
    """Run the authentication flow headless and report events via callbacks.
    Callbacks:
      - status_callback(text:str)
      - progress_callback(value:int)
      - frame_callback(frame_bgr:np.ndarray)
    
    Args:
      debug_poses (bool): if True, print detailed pose angle mismatches to console for diagnosis.
    
    Returns a dict with result info when finished.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    if len(known_encodings) > 0:
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(known_encodings)
    else:
        raise RuntimeError('No known encodings')

    authenticated_user = None
    POSE_DURATION = 3
    TOL_YAW = 15
    TOL_PITCH = 15
    TOL_ROLL = 15
    
    # Adaptive tolerances per pose: gauche/droite need wider yaw tolerance
    POSE_TOLERANCES = {
        "frontale": (TOL_YAW, TOL_PITCH, TOL_ROLL),
        "gauche": (25, TOL_PITCH, TOL_ROLL),    # wider yaw tolerance for side poses
        "droite": (25, TOL_PITCH, TOL_ROLL),    # wider yaw tolerance for side poses
        "haut": (TOL_YAW, 20, TOL_ROLL),        # wider pitch for up
        "bas": (TOL_YAW, 20, TOL_ROLL),         # wider pitch for down
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = _detector(gray, 1)
            if authenticated_user is None:
                if len(faces) > 0:
                    try:
                        enc = get_face_encoding(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), faces[0])
                        distances, indices = knn.kneighbors([enc])
                        min_dist = distances[0][0]
                        idx = indices[0][0]
                        if min_dist < 0.6:
                            authenticated_user = known_names[idx]
                            if status_callback:
                                status_callback(f"Identifié: {authenticated_user}")
                        else:
                            if status_callback:
                                status_callback("Inconnu - en attente...")
                    except Exception as e:
                        if status_callback:
                            status_callback(f"Erreur reconnaissance: {e}")
                else:
                    if status_callback:
                        status_callback("Aucun visage détecté")
            else:
                user_poses = known_poses.get(authenticated_user, {})
                poses_list = list(user_poses.keys())
                if not poses_list:
                    if status_callback:
                        status_callback("Aucune pose disponible pour l'utilisateur")
                    authenticated_user = None
                    continue
                challenge_index = 0
                while challenge_index < total_challenges:
                    required_pose = np.random.choice(poses_list)
                    if status_callback:
                        status_callback(f"Défi {challenge_index+1}/{total_challenges} : Faites la pose '{required_pose}'")
                    challenge_start = time.time()
                    hold_start = None
                    succeeded = False
                    while (time.time() - challenge_start) <= challenge_timeout:
                        ret2, frame2 = cap.read()
                        if not ret2:
                            break
                        small2 = cv2.resize(frame2, (0, 0), fx=0.75, fy=0.75)
                        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
                        faces2 = _detector(gray2, 1)
                        if len(faces2) > 0:
                            face2 = faces2[0]
                            rect_orig = dlib.rectangle(int(face2.left()/0.75), int(face2.top()/0.75), int(face2.right()/0.75), int(face2.bottom()/0.75))
                            landmarks = _predictor(frame2, rect_orig)
                            yaw, pitch, roll, _, _ = estimate_pose(landmarks, frame2.shape[1], frame2.shape[0])
                            target = user_poses.get(required_pose)
                            if target is None:
                                break
                            
                            # Use adaptive tolerances per pose
                            tol_yaw, tol_pitch, tol_roll = POSE_TOLERANCES.get(required_pose, (TOL_YAW, TOL_PITCH, TOL_ROLL))
                            
                            yaw_match = abs(yaw - target[0]) < tol_yaw
                            pitch_match = abs(pitch - target[1]) < tol_pitch
                            roll_match = abs(roll - target[2]) < tol_roll
                            
                            # DEBUG: Print pose mismatches if enabled
                            if debug_poses:
                                yaw_err = abs(yaw - target[0])
                                pitch_err = abs(pitch - target[1])
                                roll_err = abs(roll - target[2])
                                status_msg = f"{required_pose} | yaw: {yaw:.1f} vs {target[0]:.1f} (err={yaw_err:.1f}, tol={tol_yaw})"
                                status_msg += f" | pitch: {pitch:.1f} vs {target[1]:.1f} (err={pitch_err:.1f}, tol={tol_pitch})"
                                status_msg += f" | roll: {roll:.1f} vs {target[2]:.1f} (err={roll_err:.1f}, tol={tol_roll})"
                                print(status_msg)
                            
                            if yaw_match and pitch_match and roll_match:
                                if hold_start is None:
                                    hold_start = time.time()
                                elapsed_hold = time.time() - hold_start
                                progress = int(min(100, (elapsed_hold / POSE_DURATION) * 100))
                                if progress_callback:
                                    progress_callback(progress)
                                if elapsed_hold >= POSE_DURATION:
                                    succeeded = True
                                    if status_callback:
                                        status_callback(f"Défi {challenge_index+1} réussi")
                                    break
                            else:
                                hold_start = None
                                if progress_callback:
                                    progress_callback(0)
                        else:
                            hold_start = None
                            if progress_callback:
                                progress_callback(0)
                        if frame_callback:
                            frame_callback(frame2)
                        time.sleep(0.02)
                    if not succeeded:
                        if status_callback:
                            status_callback("Défi échoué : recommencer l'authentification...")
                        authenticated_user = None
                        break
                    else:
                        challenge_index += 1
                if challenge_index >= total_challenges:
                    if status_callback:
                        status_callback(f"Authentification réussie pour {authenticated_user} !")
                    if progress_callback:
                        progress_callback(100)
                    return {'result': 'success', 'user': authenticated_user}
            if frame_callback:
                frame_callback(frame)
            time.sleep(0.02)
    finally:
        cap.release()
    return {'result': 'stopped'}
