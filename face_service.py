# face_service.py — Adapté pour Microsoft Azure
import cv2
import dlib
import numpy as np
import os
import math
import time
import pickle
import hashlib
from sklearn.neighbors import NearestNeighbors

# ── Constantes ────────────────────────────────────────────────────────────────
SHAPE_PREDICTOR_PATH        = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
DIST_COEFFS                 = np.zeros((4, 1))
CACHE_FILE                  = "face_encodings_cache.pkl"

# ── Modèles dlib (chargement paresseux) ───────────────────────────────────────
_detector       = dlib.get_frontal_face_detector()
_predictor      = None
_face_recognizer = None


def _ensure_models():
    global _predictor, _face_recognizer
    if _predictor is None:
        if not os.path.exists(SHAPE_PREDICTOR_PATH):
            raise FileNotFoundError(f"shape predictor introuvable : {SHAPE_PREDICTOR_PATH}")
        _predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    if _face_recognizer is None:
        if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
            raise FileNotFoundError(f"modèle de reconnaissance introuvable : {FACE_RECOGNITION_MODEL_PATH}")
        _face_recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)


def _get_dataset_hash(dataset_dir="dataset"):
    hasher = hashlib.md5()
    if not os.path.exists(dataset_dir):
        return None
    for root, _, files in os.walk(dataset_dir):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.png')):
                path = os.path.join(root, f)
                hasher.update(path.encode())
                hasher.update(str(os.path.getmtime(path)).encode())
    return hasher.hexdigest()


# ── Estimation de pose ────────────────────────────────────────────────────────

def estimate_pose(landmarks, img_w, img_h, camera_matrix=None):
    """Retourne (yaw_deg, pitch_deg, roll_deg, rvec, tvec)."""
    if camera_matrix is None:
        camera_matrix = np.array([
            [img_w,    0,    img_w / 2.0],
            [0,        img_w, img_h / 2.0],
            [0,        0,    1.0]
        ], dtype="double")

    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype="double")

    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x,  landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0, None, None

    rot_matrix, _ = cv2.Rodrigues(rvec)
    proj_matrix   = np.hstack((rot_matrix, tvec))
    eulerAngles   = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]

    return -float(yaw[0]), float(pitch[0]), -float(roll[0]), rvec, tvec


def get_face_encoding(image_rgb, face_rect):
    _ensure_models()
    landmarks = _predictor(image_rgb, face_rect)
    return np.array(_face_recognizer.compute_face_descriptor(image_rgb, landmarks))


# ── Chargement du dataset avec cache ─────────────────────────────────────────

def load_dataset_encodings(dataset_dir="dataset", use_cache=True):
    """Charge les encodages faciaux depuis le dataset local, avec cache."""
    dataset_hash = _get_dataset_hash(dataset_dir)

    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            if cache.get("hash") == dataset_hash:
                return cache["encodings"], cache["names"], cache["poses"]
        except Exception:
            pass

    known_encodings = []
    known_names     = []
    known_poses     = {}

    if not os.path.exists(dataset_dir):
        return np.array([]), [], {}

    _ensure_models()

    for user_name in os.listdir(dataset_dir):
        user_path = os.path.join(dataset_dir, user_name)
        if not os.path.isdir(user_path):
            continue

        current_user_encodings = []
        current_user_poses_data = {
            "frontale": [], "gauche": [], "droite": [], "haut": [], "bas": []
        }

        for img_name in os.listdir(user_path):
            if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                continue
            img_path = os.path.join(user_path, img_name)
            img_bgr  = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces   = _detector(gray, 0)
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
            known_encodings.append(np.mean(current_user_encodings, axis=0))
            known_names.append(user_name)
            avg_poses = {
                k: np.mean(v, axis=0)
                for k, v in current_user_poses_data.items() if v
            }
            known_poses[user_name] = avg_poses

    enc_arr = np.array(known_encodings)

    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "hash": dataset_hash,
                "encodings": enc_arr,
                "names": known_names,
                "poses": known_poses
            }, f)
    except Exception:
        pass

    return enc_arr, known_names, known_poses


# ── Capture sans interface graphique ─────────────────────────────────────────

def capture_images_headless(user_name, num_images_per_pose=40, output_dir="dataset",
                             camera_index=0, progress_callback=None):
    """Capture d'images sans GUI. progress_callback(status, info) optionnel."""
    user_output_dir = os.path.join(output_dir, user_name)
    os.makedirs(user_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    poses = ["frontale", "gauche", "droite", "haut", "bas"]
    MIN_TIME_BETWEEN_SAVES = 0.7
    STABLE_FRAMES_REQUIRED = 3
    STABILITY_THRESH       = 20

    captured_count = 0
    for pose in poses:
        if progress_callback:
            progress_callback('pose_start', {'pose': pose})
        time.sleep(1.0)

        current_pose_count = 0
        last_save_time     = 0
        last_centers       = []

        while current_pose_count < num_images_per_pose:
            ret, frame = cap.read()
            if not ret:
                break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _detector(gray)
            if len(faces) == 1:
                fr = faces[0]
                cx = fr.left() + fr.width() / 2
                cy = fr.top()  + fr.height() / 2
                last_centers.append((cx, cy))
                if len(last_centers) > STABLE_FRAMES_REQUIRED:
                    last_centers.pop(0)

                stable = False
                if len(last_centers) == STABLE_FRAMES_REQUIRED:
                    max_dist = 0
                    for i in range(len(last_centers)):
                        for j in range(i + 1, len(last_centers)):
                            dist = math.hypot(last_centers[i][0] - last_centers[j][0],
                                              last_centers[i][1] - last_centers[j][1])
                            if dist > max_dist:
                                max_dist = dist
                    if max_dist <= STABILITY_THRESH:
                        stable = True

                now = time.time()
                if stable and (now - last_save_time >= MIN_TIME_BETWEEN_SAVES):
                    img_path = os.path.join(user_output_dir,
                                            f"{user_name}_{pose}_{current_pose_count:03d}.jpg")
                    cv2.imwrite(img_path, frame)
                    current_pose_count += 1
                    captured_count     += 1
                    last_save_time      = now
                    if progress_callback:
                        progress_callback('image_saved', {
                            'pose': pose, 'count': current_pose_count, 'path': img_path
                        })
            else:
                last_centers = []

        if progress_callback:
            progress_callback('pose_done', {'pose': pose})
        time.sleep(1.0)

    cap.release()
    return {'user': user_name, 'total_captured': captured_count}


# ── Authentification headless ─────────────────────────────────────────────────

def authenticate_sequence(known_encodings, known_names, known_poses,
                           camera_index=0, total_challenges=5, challenge_timeout=12,
                           progress_callback=None, status_callback=None,
                           frame_callback=None, debug_poses=False):
    """
    Exécute le flux d'authentification complet sans GUI.
    Retourne un dict {'result': 'success'/'stopped', 'user': ...}
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    if len(known_encodings) > 0:
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(known_encodings)
    else:
        raise RuntimeError("Aucun encodage connu")

    authenticated_user = None
    POSE_DURATION      = 3
    TOL_YAW            = 15
    TOL_PITCH          = 15
    TOL_ROLL           = 15

    POSE_TOLERANCES = {
        "frontale": (TOL_YAW, TOL_PITCH, TOL_ROLL),
        "gauche":   (25, TOL_PITCH, TOL_ROLL),
        "droite":   (25, TOL_PITCH, TOL_ROLL),
        "haut":     (TOL_YAW, 20, TOL_ROLL),
        "bas":      (TOL_YAW, 20, TOL_ROLL),
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = _detector(gray, 1)

            if authenticated_user is None:
                if len(faces) > 0:
                    try:
                        enc = get_face_encoding(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), faces[0])
                        distances, indices = knn.kneighbors([enc])
                        if distances[0][0] < 0.6:
                            authenticated_user = known_names[indices[0][0]]
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
                        status_callback("Aucune pose disponible")
                    authenticated_user = None
                    continue

                challenge_index = 0
                while challenge_index < total_challenges:
                    required_pose  = np.random.choice(poses_list)
                    if status_callback:
                        status_callback(f"Défi {challenge_index+1}/{total_challenges} : Faites la pose '{required_pose}'")

                    challenge_start = time.time()
                    hold_start      = None
                    succeeded       = False

                    while (time.time() - challenge_start) <= challenge_timeout:
                        ret2, frame2 = cap.read()
                        if not ret2:
                            break
                        small2 = cv2.resize(frame2, (0, 0), fx=0.75, fy=0.75)
                        gray2  = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
                        faces2 = _detector(gray2, 1)

                        if len(faces2) > 0:
                            face2 = faces2[0]
                            rect  = dlib.rectangle(
                                int(face2.left() / 0.75), int(face2.top() / 0.75),
                                int(face2.right() / 0.75), int(face2.bottom() / 0.75)
                            )
                            landmarks = _predictor(frame2, rect)
                            yaw, pitch, roll, _, _ = estimate_pose(
                                landmarks, frame2.shape[1], frame2.shape[0]
                            )
                            target = user_poses.get(required_pose)
                            if target is None:
                                break

                            tol_yaw, tol_pitch, tol_roll = POSE_TOLERANCES.get(
                                required_pose, (TOL_YAW, TOL_PITCH, TOL_ROLL)
                            )

                            if debug_poses:
                                print(f"{required_pose} | yaw: {yaw:.1f} vs {target[0]:.1f} "
                                      f"(err={abs(yaw-target[0]):.1f}, tol={tol_yaw}) | "
                                      f"pitch: {pitch:.1f} vs {target[1]:.1f} "
                                      f"(err={abs(pitch-target[1]):.1f}, tol={tol_pitch})")

                            if abs(yaw - target[0]) < tol_yaw and abs(pitch - target[1]) < tol_pitch:
                                if hold_start is None:
                                    hold_start = time.time()
                                elapsed = time.time() - hold_start
                                if progress_callback:
                                    progress_callback(int(min(100, (elapsed / POSE_DURATION) * 100)))
                                if elapsed >= POSE_DURATION:
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
