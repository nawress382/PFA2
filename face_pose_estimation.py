# face_pose_estimation.py — Adapté pour Microsoft Azure
import cv2
import numpy as np
import os
import math
import time
import re
import threading
import pickle
from sklearn.neighbors import NearestNeighbors
import sys

# ── Chargement du .env ────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
# ──────────────────────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog, QProgressBar, QGraphicsDropShadowEffect, QDialog, QFrame, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont, QPainter, QPen, QLinearGradient, QBrush, QKeySequence, QShortcut
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QRect, pyqtProperty
import uuid
from trust_qr_module import TrustQRDialog, generate_trust_code

# ── Azure SDK ──────────────────────────────────────────────────────────────────
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from security_utils import encrypt_aes_512, upload_to_azure_blob

# ── Import de l'interface de migration ────────────────────────────────────────
from migrate_gui import MigrateApp

# ── Import des fonctions de service ───────────────────────────────────────────
from face_service import (
    estimate_pose as service_estimate_pose,
    get_face_encoding as service_get_face_encoding,
    load_dataset_encodings as service_load_dataset_encodings,
    capture_images_headless as service_capture_images_headless,
    authenticate_sequence as service_authenticate_sequence,
)

# ── Synchronisation du dataset avec Azure ─────────────────────────────────────
from dataset_sync import sync_dataset_on_startup, upload_dataset_to_azure

# ── Configuration Azure Cosmos DB ─────────────────────────────────────────────
COSMOS_ENDPOINT  = os.environ.get("COSMOS_ENDPOINT", "https://pfa2-cosmos-auth.documents.azure.com:443/")
COSMOS_KEY       = os.environ.get("COSMOS_KEY", "")
COSMOS_DB_NAME   = "FaceAuthDB"
COSMOS_CONTAINER = "AuthTasks"

# ── Configuration Azure Blob Storage ──────────────────────────────────────────
AZURE_CONTAINER_DOCS = "documents-chiffres"
AZURE_CONTAINER_LOGS = "logs-audit"


def _get_cosmos_container():
    try:
        if COSMOS_KEY:
            client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
        else:
            credential = DefaultAzureCredential()
            client = CosmosClient(COSMOS_ENDPOINT, credential=credential)
        db        = client.get_database_client(COSMOS_DB_NAME)
        container = db.get_container_client(COSMOS_CONTAINER)
        return container
    except Exception as e:
        print(f"[Cosmos DB] Erreur connexion: {e}")
        return None


def _cosmos_upsert(item: dict):
    container = _get_cosmos_container()
    if container is None:
        return
    try:
        if 'id' not in item:
            item['id'] = item.get('task_id', str(uuid.uuid4()))
        if 'user_id' not in item:
            item['user_id'] = item.get('user', 'unknown')
        container.upsert_item(item)
    except Exception as e:
        print(f"[Cosmos DB] Erreur upsert: {e}")


def _cosmos_update_status(task_id: str, user_id: str, updates: dict):
    container = _get_cosmos_container()
    if container is None:
        return
    try:
        item = container.read_item(item=task_id, partition_key=user_id)
        item.update(updates)
        container.replace_item(item=task_id, body=item)
    except Exception as e:
        print(f"[Cosmos DB] Erreur update: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PALETTE & STYLES
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "bg":           "#F7F8FC",
    "sidebar":      "#1A2340",
    "card":         "#FFFFFF",
    "accent":       "#2563EB",
    "accent_light": "#EFF6FF",
    "success":      "#059669",
    "danger":       "#DC2626",
    "text_main":    "#0F172A",
    "text_sub":     "#64748B",
    "text_white":   "#F1F5F9",
    "border":       "#E2E8F0",
    "progress_bg":  "#DBEAFE",
    "separator":    "#CBD5E1",
}

STYLESHEET_MAIN = f"""
QWidget#mainWindow {{
    background-color: {PALETTE['bg']};
    font-family: 'Segoe UI', 'Calibri', sans-serif;
}}
QFrame#sidebar {{
    background-color: {PALETTE['sidebar']};
    border-radius: 0px;
}}
QLabel#appTitle {{
    color: {PALETTE['text_white']};
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#appSubtitle {{
    color: #94A3B8;
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 2px;
}}
QLabel#logoIcon {{
    color: {PALETTE['accent']};
    font-size: 36px;
}}
QFrame#userCard {{
    background-color: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(37, 99, 235, 0.3);
    border-radius: 12px;
}}
QLabel#userLabel {{
    color: #94A3B8;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
}}
QLabel#userName {{
    color: {PALETTE['text_white']};
    font-size: 16px;
    font-weight: 700;
}}
QLabel#userAvatar {{
    color: {PALETTE['accent']};
    font-size: 28px;
}}
QLabel#statusBadge {{
    color: #94A3B8;
    font-size: 12px;
    font-weight: 400;
}}
QPushButton#btnCapture {{
    background-color: {PALETTE['accent']};
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    padding: 12px 8px;
    text-align: left;
}}
QPushButton#btnCapture:hover {{ background-color: #1D4ED8; }}
QPushButton#btnAuth {{
    background-color: rgba(37, 99, 235, 0.12);
    color: {PALETTE['text_white']};
    border: 1px solid rgba(37, 99, 235, 0.4);
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    padding: 12px 8px;
    text-align: left;
}}
QPushButton#btnAuth:hover {{ background-color: rgba(37, 99, 235, 0.25); }}
QPushButton#btnQuit {{
    background-color: transparent;
    color: #94A3B8;
    border: 1px solid #334155;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 500;
    padding: 10px 8px;
    text-align: left;
}}
QPushButton#btnQuit:hover {{
    background-color: rgba(220, 38, 38, 0.15);
    color: #FCA5A5;
    border-color: rgba(220, 38, 38, 0.4);
}}
QFrame#cameraCard {{
    background-color: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 20px;
}}
QLabel#cameraTitle {{
    color: {PALETTE['text_main']};
    font-size: 15px;
    font-weight: 700;
}}
QLabel#cameraSubtitle {{
    color: {PALETTE['text_sub']};
    font-size: 12px;
}}
QLabel#videoLabel {{
    background-color: #0F172A;
    border-radius: 14px;
}}
QProgressBar#authProgress {{
    background-color: {PALETTE['progress_bg']};
    border: none;
    border-radius: 6px;
    height: 10px;
}}
QProgressBar#authProgress::chunk {{
    background-color: qlineargradient(
        spread:pad, x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563EB, stop:1 #60A5FA
    );
    border-radius: 6px;
}}
QLabel#progressLabel {{
    color: {PALETTE['accent']};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel#challengeBadge {{
    background-color: {PALETTE['accent_light']};
    color: {PALETTE['accent']};
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
}}
"""

STYLESHEET_DIALOG = f"""
QDialog#captureDialog {{
    background-color: {PALETTE['bg']};
    font-family: 'Segoe UI', 'Calibri', sans-serif;
}}
QLabel#dlgTitle {{
    color: {PALETTE['text_main']};
    font-size: 18px;
    font-weight: 700;
}}
QLabel#dlgSubtitle {{
    color: {PALETTE['text_sub']};
    font-size: 12px;
}}
QLabel#dlgInstruction {{
    color: {PALETTE['accent']};
    background-color: {PALETTE['accent_light']};
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    padding: 10px 16px;
}}
QLabel#dlgStatus {{
    color: {PALETTE['text_sub']};
    font-size: 12px;
}}
QLabel#dlgVideoLabel {{
    background-color: #0F172A;
    border-radius: 12px;
}}
QProgressBar#dlgProgress {{
    background-color: {PALETTE['progress_bg']};
    border: none;
    border-radius: 5px;
    height: 8px;
}}
QProgressBar#dlgProgress::chunk {{
    background-color: {PALETTE['accent']};
    border-radius: 5px;
}}
QPushButton#dlgOk {{
    background-color: {PALETTE['success']};
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 24px;
}}
QPushButton#dlgOk:hover {{ background-color: #047857; }}
QPushButton#dlgCancel {{
    background-color: transparent;
    color: {PALETTE['danger']};
    border: 1px solid {PALETTE['danger']};
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 24px;
}}
QPushButton#dlgCancel:hover {{
    background-color: rgba(220, 38, 38, 0.08);
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

SHAPE_PREDICTOR_PATH        = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
DATASET_DIR                 = "dataset"
FACE_DETECTION_THRESHOLD    = 0.6
CACHE_PATH                  = "face_encodings_cache.pkl"

MIN_TIME_BETWEEN_SAVES  = 0.7
WAIT_BETWEEN_POSES      = 5
STABLE_FRAMES_REQUIRED  = 3
STABILITY_THRESH        = 20
TOTAL_CHALLENGES        = 2
CHALLENGE_TIMEOUT       = 12

# ── dlib chargé dans DlibLoaderThread — PAS à l'import du module ─────────────
detector        = None
predictor       = None
face_recognizer = None

CAMERA_MATRIX = np.array([[0.0, 0, 0], [0, 0.0, 0], [0, 0, 1]], dtype="double")
DIST_COEFFS   = np.zeros((4, 1))

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype="double")


def get_face_encoding(image_rgb, face_rect):
    landmarks = predictor(image_rgb, face_rect)
    return np.array(face_recognizer.compute_face_descriptor(image_rgb, landmarks))


def estimate_pose(landmarks, img_w, img_h):
    global CAMERA_MATRIX
    CAMERA_MATRIX[0, 0] = img_w
    CAMERA_MATRIX[1, 1] = img_w
    CAMERA_MATRIX[0, 2] = img_w / 2
    CAMERA_MATRIX[1, 2] = img_h / 2

    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x,  landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")

    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0, 0, 0, None, None

    rot_matrix, _ = cv2.Rodrigues(rvec)
    proj_matrix   = np.hstack((rot_matrix, tvec))
    eulerAngles   = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]
    return -yaw[0], pitch[0], -roll[0], rvec, tvec


# ══════════════════════════════════════════════════════════════════════════════
# THREADS
# ══════════════════════════════════════════════════════════════════════════════

class DlibLoaderThread(QThread):
    finished      = pyqtSignal()
    status_update = pyqtSignal(str)
    error         = pyqtSignal(str)

    def run(self):
        global detector, predictor, face_recognizer
        try:
            import dlib as _dlib

            self.status_update.emit("⏳  Chargement détecteur de visages...")
            detector = _dlib.get_frontal_face_detector()

            self.status_update.emit("⏳  Chargement modèle landmarks (95 MB)...")
            if not os.path.exists(SHAPE_PREDICTOR_PATH):
                self.error.emit(f"❌  Fichier introuvable : {SHAPE_PREDICTOR_PATH}")
                return
            predictor = _dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

            self.status_update.emit("⏳  Chargement modèle reconnaissance (21 MB)...")
            if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
                self.error.emit(f"❌  Fichier introuvable : {FACE_RECOGNITION_MODEL_PATH}")
                return
            face_recognizer = _dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

            self.status_update.emit("✅  Modèles IA chargés")
        except Exception as e:
            self.error.emit(f"❌  Erreur chargement dlib : {e}")
        finally:
            self.finished.emit()


class DatasetLoaderThread(QThread):
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent

    def run(self):
        try:
            print("\n[Dataset Loader] Démarrage...")

            def _sync_in_background():
                try:
                    sync_msg = sync_dataset_on_startup()
                    print(f"[Dataset Sync] {sync_msg}")
                except Exception as e:
                    print(f"[Dataset Sync] Avertissement: {e}")
            threading.Thread(target=_sync_in_background, daemon=True).start()

            # Vérification validité du cache
            nb_images = sum(len(files) for _, _, files in os.walk("dataset")) if os.path.exists("dataset") else 0

            cache_valid = False
            if nb_images == 0:
                if os.path.exists(CACHE_PATH):
                    os.remove(CACHE_PATH)
                    print("[Cache] Dataset vide — cache supprimé")
            elif os.path.exists(CACHE_PATH):
                cache_time   = os.path.getmtime(CACHE_PATH)
                dataset_time = 0
                for root, dirs, files in os.walk("dataset"):
                    for f in files:
                        t = os.path.getmtime(os.path.join(root, f))
                        if t > dataset_time:
                            dataset_time = t
                if cache_time >= dataset_time:
                    cache_valid = True

            if cache_valid:
                print("[Dataset Loader] Cache valide — chargement rapide...")
                with open(CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                encodings = data["encodings"]
                names     = data["names"]
                poses     = data["poses"]
                print(f"[Dataset Loader] {len(names)} utilisateur(s) chargé(s) depuis cache")
            else:
                print("[Dataset Loader] Calcul des encodages...")
                encodings, names, poses = service_load_dataset_encodings()
                with open(CACHE_PATH, "wb") as f:
                    pickle.dump({"encodings": encodings, "names": names, "poses": poses}, f)
                print(f"[Dataset Loader] Cache sauvegardé — {len(names)} utilisateur(s)")

            self.parent_app.known_encodings = encodings
            self.parent_app.known_names     = names
            self.parent_app.known_poses     = poses

            print(f"\n=== DIAGNOSTIC POSES APPRISES ===")
            for user, poses_dict in poses.items():
                print(f"\n{user}:")
                for pose_name, (yaw, pitch, roll) in poses_dict.items():
                    print(f"  {pose_name}: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
            print(f"===================================\n")

        except Exception as e:
            print(f"[Dataset Loader] Erreur: {e}")
        finally:
            self.finished.emit()


class CaptureThread(QThread):
    frame_ready     = pyqtSignal(QImage)
    image_captured  = pyqtSignal(str, str)
    status_update   = pyqtSignal(str)
    progress_update = pyqtSignal(int)

    def __init__(self, user_name, poses_list, num_per_pose=40, parent=None):
        super().__init__(parent)
        self.user_name        = user_name
        self.poses_list       = poses_list
        self.num_per_pose     = num_per_pose
        self._running         = True
        self.captured_count   = 0
        self.total_to_capture = len(poses_list) * num_per_pose
        self.task_id          = f"CAP-{user_name}-{int(time.time())}"

    def run(self):
        _cosmos_upsert({
            'task_id':   self.task_id,
            'status':    'capture_starting',
            'user':      self.user_name,
            'user_id':   self.user_name,
            'progress':  0,
            'timestamp': str(time.time())
        })

        user_output_dir = os.path.join("dataset", self.user_name)
        os.makedirs(user_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_update.emit("Erreur : Impossible d'ouvrir la caméra")
            return

        try:
            for pose_idx, pose in enumerate(self.poses_list):
                if not self._running:
                    break

                self.status_update.emit(f"Préparation pour la pose : {pose}")
                _cosmos_update_status(self.task_id, self.user_name, {'status': f'capturing_pose_{pose}'})

                for i in range(WAIT_BETWEEN_POSES):
                    if not self._running:
                        break
                    self.status_update.emit(f"Pose '{pose}' commence dans {WAIT_BETWEEN_POSES - i}s...")
                    time.sleep(1)

                current_pose_count = 0
                last_save_time     = 0
                last_centers       = []

                while current_pose_count < self.num_per_pose and self._running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    display_frame = frame.copy()
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    if len(faces) == 1:
                        face_rect = faces[0]
                        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cx, cy = x + w / 2, y + h / 2
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
                            img_path = os.path.join(user_output_dir, f"{self.user_name}_{pose}_{current_pose_count:03d}.jpg")
                            cv2.imwrite(img_path, frame)
                            current_pose_count  += 1
                            self.captured_count += 1
                            last_save_time       = now
                            self.image_captured.emit(pose, img_path)
                            prog = int((self.captured_count / self.total_to_capture) * 100)
                            self.progress_update.emit(prog)

                        self.status_update.emit(f"Pose {pose}: {current_pose_count}/{self.num_per_pose}")
                    else:
                        self.status_update.emit("Placez votre visage au centre")

                    rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    qt_img = QImage(rgb_display.data, rgb_display.shape[1], rgb_display.shape[0],
                                    rgb_display.shape[1] * 3, QImage.Format.Format_RGB888)
                    self.frame_ready.emit(qt_img)
                    time.sleep(0.03)

                if pose_idx < len(self.poses_list) - 1:
                    time.sleep(3)

        finally:
            cap.release()
            if self._running:
                if os.path.exists(CACHE_PATH):
                    os.remove(CACHE_PATH)
                    print("[Cache] Cache invalidé — recalcul au prochain démarrage")

                self.status_update.emit("Capture terminée — Upload vers Azure...")
                try:
                    result = upload_dataset_to_azure(
                        dataset_dir="dataset",
                        progress_callback=lambda s, i: print(f"[Upload] {i}")
                    )
                    self.status_update.emit(
                        f"✓ Dataset synchronisé sur Azure ({result['uploaded']} images)"
                    )
                except Exception as e:
                    self.status_update.emit(f"⚠ Capture OK mais erreur upload Azure: {e}")

                _cosmos_update_status(self.task_id, self.user_name, {
                    'status': 'capture_completed_locally',
                    'progress': 100
                })
            else:
                self.status_update.emit("Capture annulée")

    def stop(self):
        self._running = False
        self.wait()


# ══════════════════════════════════════════════════════════════════════════════
# WORKER D'AUTHENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class AuthenticationWorker(QThread):
    frame_ready     = pyqtSignal(QImage)
    status_update   = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    finished        = pyqtSignal()

    def __init__(self, known_encodings, known_names, known_poses, parent=None):
        super().__init__(parent)
        self.known_encodings = known_encodings
        self.known_names     = known_names
        self.known_poses     = known_poses
        self._running        = True

    def run(self):
        import dlib as _dlib

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_update.emit("Erreur : Impossible d'ouvrir la caméra.")
            self.finished.emit()
            return

        if len(self.known_encodings) > 0:
            face_recognizer_knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            face_recognizer_knn.fit(self.known_encodings)
        else:
            self.status_update.emit("Aucun encodage facial enregistré.")
            cap.release()
            self.finished.emit()
            return

        POSE_DURATION            = 3
        authenticated_user       = None
        authentication_succeeded = False
        task_id = str(uuid.uuid4())

        try:
            while self._running and not authentication_succeeded:
                ret, frame = cap.read()
                if not ret:
                    self.status_update.emit("Erreur lecture caméra")
                    break

                small      = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                rgb_small  = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces      = detector(gray_small, 1)

                if authenticated_user is None:
                    if len(faces) > 0:
                        try:
                            enc = get_face_encoding(rgb_small, faces[0])
                            distances, indices = face_recognizer_knn.kneighbors([enc])
                            min_distance = distances[0][0]
                            idx          = indices[0][0]
                            if min_distance < FACE_DETECTION_THRESHOLD:
                                authenticated_user = self.known_names[idx]
                                self.status_update.emit(f"Identifié: {authenticated_user}")
                                _cosmos_upsert({
                                    'task_id':  task_id,
                                    'status':   'user_identified',
                                    'user':     authenticated_user,
                                    'user_id':  authenticated_user,
                                    'progress': 10
                                })
                                user_poses = self.known_poses.get(authenticated_user)
                                if not user_poses or len(user_poses) == 0:
                                    self.status_update.emit("Aucune pose enregistrée")
                                    authenticated_user = None
                            else:
                                self.status_update.emit("Inconnu - en attente...")
                        except Exception as e:
                            self.status_update.emit(f"Erreur reconnaissance: {e}")
                    else:
                        self.status_update.emit("Aucun visage détecté")

                else:
                    user_poses  = self.known_poses.get(authenticated_user, {})
                    poses_list  = list(user_poses.keys())

                    challenge_index = 0
                    while challenge_index < TOTAL_CHALLENGES and self._running:
                        required_pose = np.random.choice(poses_list)
                        self.status_update.emit(f"Défi {challenge_index+1}/{TOTAL_CHALLENGES} : {required_pose}")

                        challenge_start = time.time()
                        hold_start      = None
                        succeeded       = False

                        while (time.time() - challenge_start) <= CHALLENGE_TIMEOUT and self._running:
                            ret2, frame2 = cap.read()
                            if not ret2:
                                break

                            small2 = cv2.resize(frame2, (0, 0), fx=0.75, fy=0.75)
                            gray2  = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
                            faces2 = detector(gray2, 1)

                            if len(faces2) > 0:
                                face2 = faces2[0]
                                face_rect_orig = _dlib.rectangle(
                                    int(face2.left() / 0.75), int(face2.top() / 0.75),
                                    int(face2.right() / 0.75), int(face2.bottom() / 0.75)
                                )
                                landmarks = predictor(frame2, face_rect_orig)
                                yaw, pitch, roll, rvec, tvec = estimate_pose(
                                    landmarks, frame2.shape[1], frame2.shape[0]
                                )

                                target = user_poses.get(required_pose)
                                if target is None:
                                    break

                                if abs(yaw - target[0]) < 20 and abs(pitch - target[1]) < 20:
                                    if hold_start is None:
                                        hold_start = time.time()
                                    elapsed = time.time() - hold_start
                                    self.progress_update.emit(int(min(100, (elapsed / POSE_DURATION) * 100)))
                                    if elapsed >= POSE_DURATION:
                                        succeeded = True
                                        break
                                else:
                                    hold_start = None
                                    self.progress_update.emit(0)

                            rgb_disp = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                            qimg = QImage(rgb_disp.data, frame2.shape[1], frame2.shape[0],
                                          frame2.shape[1] * 3, QImage.Format.Format_RGB888)
                            self.frame_ready.emit(qimg)
                            time.sleep(0.02)

                        if not succeeded:
                            self.status_update.emit("Défi échoué")
                            authenticated_user = None
                            break
                        else:
                            challenge_index += 1
                            _cosmos_update_status(task_id, authenticated_user, {'progress': 50})

                    if challenge_index >= TOTAL_CHALLENGES:
                        self.status_update.emit("Succès ! Génération du Code de Confiance...")

                        session_data = generate_trust_code(authenticated_user)
                        trust_code   = session_data["trust_code"]
                        task_id      = session_data["task_id"]

                        _cosmos_upsert({
                            'id':         task_id,
                            'task_id':    task_id,
                            'trust_code': trust_code,
                            'status':     'trust_code_generated',
                            'user':       authenticated_user,
                            'user_id':    authenticated_user,
                            'progress':   75,
                            'timestamp':  str(time.time())
                        })

                        self.status_update.emit(
                            f"TRUST_CODE_READY:{authenticated_user}:{task_id}"
                        )

                        time.sleep(3.0)

                        try:
                            secret_data = (
                                f"Auth réussie pour {authenticated_user} | "
                                f"Code: {trust_code} | "
                                f"Session: {task_id} | "
                                f"Date: {time.ctime()}"
                            ).encode()

                            payload, signature = encrypt_aes_512(secret_data)
                            blob_path = f"logs_authentification/{authenticated_user}_{int(time.time())}.bin"

                            success = upload_to_azure_blob(
                                payload, signature,
                                AZURE_CONTAINER_LOGS,
                                blob_path
                            )

                            if success:
                                _cosmos_update_status(task_id, authenticated_user, {
                                    'status':    'finished_and_migrated',
                                    'progress':  100,
                                    'blob_path': blob_path
                                })
                                self.status_update.emit("✓ authentifié et sécurisé sur Azure")
                            else:
                                self.status_update.emit("⚠ Authentifié mais erreur upload Azure")

                        except Exception as e:
                            self.status_update.emit(f"Erreur Cloud Azure: {e}")

                        time.sleep(1.0)
                        authentication_succeeded = True
                        break

                time.sleep(0.02)

        finally:
            cap.release()
            self.progress_update.emit(0)
            self.status_update.emit("Session terminée")
            self.finished.emit()


# ══════════════════════════════════════════════════════════════════════════════
# CAPTURE DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class CaptureDialog(QDialog):
    def __init__(self, user_name, parent=None):
        super().__init__(parent)
        self.setObjectName("captureDialog")
        self.setWindowTitle(f"Enregistrement — {user_name}")
        self.setFixedSize(680, 780)
        self.setStyleSheet(STYLESHEET_DIALOG)

        layout = QVBoxLayout()
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        header = QHBoxLayout()
        icon = QLabel("📸")
        icon.setStyleSheet("font-size: 30px;")
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("Enregistrement facial")
        title.setObjectName("dlgTitle")
        sub = QLabel(f"Utilisateur : {user_name}")
        sub.setObjectName("dlgSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        header.addWidget(icon)
        header.addSpacing(10)
        header.addLayout(title_col)
        header.addStretch()
        layout.addLayout(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {PALETTE['border']}; max-height: 1px;")
        layout.addWidget(sep)

        self.instruction_label = QLabel("Initialisation de la caméra...")
        self.instruction_label.setObjectName("dlgInstruction")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)

        self.video_label = QLabel()
        self.video_label.setObjectName("dlgVideoLabel")
        self.video_label.setFixedSize(500, 400)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(37, 99, 235, 60))
        shadow.setOffset(0, 8)
        self.video_label.setGraphicsEffect(shadow)
        video_row = QHBoxLayout()
        video_row.addStretch()
        video_row.addWidget(self.video_label)
        video_row.addStretch()
        layout.addLayout(video_row)

        prog_label = QLabel("PROGRESSION DE LA CAPTURE")
        prog_label.setStyleSheet(f"color: {PALETTE['text_sub']}; font-size: 10px; font-weight: 600; letter-spacing: 1.5px;")
        layout.addWidget(prog_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("dlgProgress")
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Veuillez vous placer face à l'objectif")
        self.status_label.setObjectName("dlgStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addStretch()
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self.cancel_button = QPushButton("✕  Annuler")
        self.cancel_button.setObjectName("dlgCancel")
        self.ok_button = QPushButton("✓  Terminer")
        self.ok_button.setObjectName("dlgOk")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.cancel_button)
        btn_row.addWidget(self.ok_button)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def update_pose_status(self, current_pose):
        self.instruction_label.setText(f"Regardez vers : {current_pose.upper()}")

    def set_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        ))


# ══════════════════════════════════════════════════════════════════════════════
# FENÊTRE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class FaceAuthApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("mainWindow")
        self.setWindowTitle("Authentification Faciale — Azure Cloud")
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(STYLESHEET_MAIN)

        self.known_encodings = np.array([])
        self.known_names     = []
        self.known_poses     = {}
        self.auth_worker     = None
        self._ring_angle     = 0.0
        self._scan_pos       = 0.0
        self._scan_dir       = 1

        esc = QShortcut(QKeySequence("Escape"), self)
        esc.activated.connect(self._exit_fullscreen)

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── SIDEBAR ───────────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(24, 32, 24, 28)
        sidebar_layout.setSpacing(0)

        logo_row = QHBoxLayout()
        logo_icon = QLabel("🔐")
        logo_icon.setObjectName("logoIcon")
        logo_text_col = QVBoxLayout()
        logo_text_col.setSpacing(1)
        app_title = QLabel("FACEAUTH")
        app_title.setObjectName("appTitle")
        app_sub = QLabel("AZURE CLOUD")
        app_sub.setObjectName("appSubtitle")
        logo_text_col.addWidget(app_title)
        logo_text_col.addWidget(app_sub)
        logo_row.addWidget(logo_icon)
        logo_row.addSpacing(10)
        logo_row.addLayout(logo_text_col)
        logo_row.addStretch()
        sidebar_layout.addLayout(logo_row)
        sidebar_layout.addSpacing(32)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("background-color: #273354; max-height: 1px;")
        sidebar_layout.addWidget(sep1)
        sidebar_layout.addSpacing(24)

        self.user_card = QFrame()
        self.user_card.setObjectName("userCard")
        user_card_layout = QHBoxLayout()
        user_card_layout.setContentsMargins(14, 14, 14, 14)
        user_card_layout.setSpacing(10)
        self.user_avatar = QLabel("👤")
        self.user_avatar.setObjectName("userAvatar")
        user_info_col = QVBoxLayout()
        user_info_col.setSpacing(2)
        user_lbl = QLabel("UTILISATEUR IDENTIFIÉ")
        user_lbl.setObjectName("userLabel")
        self.user_name_label = QLabel("—")
        self.user_name_label.setObjectName("userName")
        user_info_col.addWidget(user_lbl)
        user_info_col.addWidget(self.user_name_label)
        user_card_layout.addWidget(self.user_avatar)
        user_card_layout.addLayout(user_info_col)
        user_card_layout.addStretch()
        self.user_card.setLayout(user_card_layout)
        sidebar_layout.addWidget(self.user_card)
        sidebar_layout.addSpacing(28)

        nav_lbl = QLabel("ACTIONS")
        nav_lbl.setStyleSheet("color: #475569; font-size: 10px; font-weight: 600; letter-spacing: 2px;")
        sidebar_layout.addWidget(nav_lbl)
        sidebar_layout.addSpacing(10)

        self.capture_button = QPushButton("  📷   Capturer Images")
        self.capture_button.setObjectName("btnCapture")
        self.capture_button.setFixedHeight(46)

        self.auth_button = QPushButton("  🔍   Authentifier")
        self.auth_button.setObjectName("btnAuth")
        self.auth_button.setFixedHeight(46)

        self.fullscreen_button = QPushButton("  ⛶   Plein écran")
        self.fullscreen_button.setObjectName("btnAuth")
        self.fullscreen_button.setFixedHeight(46)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)

        sidebar_layout.addWidget(self.capture_button)
        sidebar_layout.addSpacing(8)
        sidebar_layout.addWidget(self.auth_button)
        sidebar_layout.addSpacing(8)
        sidebar_layout.addWidget(self.fullscreen_button)
        sidebar_layout.addStretch()

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #273354; max-height: 1px;")
        sidebar_layout.addWidget(sep2)
        sidebar_layout.addSpacing(16)

        self.status_label = QLabel("⏳  Initialisation...")
        self.status_label.setObjectName("statusBadge")
        self.status_label.setWordWrap(True)
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addSpacing(16)

        self.quit_button = QPushButton("  ✕   Quitter l'application")
        self.quit_button.setObjectName("btnQuit")
        self.quit_button.setFixedHeight(42)
        sidebar_layout.addWidget(self.quit_button)

        sidebar.setLayout(sidebar_layout)
        root.addWidget(sidebar)

        # ── ZONE DROITE ───────────────────────────────────────────────────────
        right_area = QVBoxLayout()
        right_area.setContentsMargins(28, 28, 28, 28)
        right_area.setSpacing(16)

        header_row = QHBoxLayout()
        cam_title = QLabel("Flux Caméra en Direct")
        cam_title.setObjectName("cameraTitle")
        cam_sub = QLabel("Détection et vérification biométrique faciale")
        cam_sub.setObjectName("cameraSubtitle")
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_col.addWidget(cam_title)
        title_col.addWidget(cam_sub)
        header_row.addLayout(title_col)
        header_row.addStretch()
        self.challenge_badge = QLabel("")
        self.challenge_badge.setObjectName("challengeBadge")
        self.challenge_badge.hide()
        header_row.addWidget(self.challenge_badge)
        right_area.addLayout(header_row)

        camera_card = QFrame()
        camera_card.setObjectName("cameraCard")
        camera_card_layout = QVBoxLayout()
        camera_card_layout.setContentsMargins(20, 20, 20, 20)
        camera_card_layout.setSpacing(14)

        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(900, 580)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vid_shadow = QGraphicsDropShadowEffect()
        vid_shadow.setBlurRadius(20)
        vid_shadow.setColor(QColor(0, 0, 0, 80))
        vid_shadow.setOffset(0, 4)
        self.video_label.setGraphicsEffect(vid_shadow)
        vid_row = QHBoxLayout()
        vid_row.addStretch()
        vid_row.addWidget(self.video_label)
        vid_row.addStretch()
        camera_card_layout.addLayout(vid_row)

        prog_header = QHBoxLayout()
        prog_lbl = QLabel("PROGRESSION AUTHENTIFICATION")
        prog_lbl.setObjectName("progressLabel")
        self.prog_pct = QLabel("0%")
        self.prog_pct.setObjectName("progressLabel")
        prog_header.addWidget(prog_lbl)
        prog_header.addStretch()
        prog_header.addWidget(self.prog_pct)
        camera_card_layout.addLayout(prog_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("authProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        camera_card_layout.addWidget(self.progress_bar)

        camera_card.setLayout(camera_card_layout)
        right_area.addWidget(camera_card)
        right_area.addStretch()

        root.addLayout(right_area)
        self.setLayout(root)

        self.capture_button.clicked.connect(self.capture_images)
        self.auth_button.clicked.connect(self.authenticate)
        self.quit_button.clicked.connect(self.close_app)

        self.capture_button.setEnabled(False)
        self.auth_button.setEnabled(False)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.dlib_loader = DlibLoaderThread(self)
        self.dlib_loader.status_update.connect(self.status_label.setText)
        self.dlib_loader.error.connect(self.status_label.setText)
        self.dlib_loader.finished.connect(self._on_dlib_loaded)
        self.dlib_loader.start()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("  ⛶   Plein écran")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("  ⊡   Fenêtré")

    def _exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("  ⛶   Plein écran")

    def _on_dlib_loaded(self):
        if detector is None:
            self.status_label.setText("❌  Erreur chargement modèles IA")
            return
        self.capture_button.setEnabled(True)
        self.auth_button.setEnabled(True)
        self.dataset_loader = DatasetLoaderThread(self)
        self.dataset_loader.finished.connect(self._on_dataset_loaded)
        self.dataset_loader.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        lw = self.video_label.width()
        lh = self.video_label.height()

        scale   = min(lw / w, lh / h)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rh, rw, _ = resized.shape
        overlay = resized.copy()
        center  = (rw // 2, rh // 2)
        radius  = int(min(rw, rh) * 0.36)

        cv2.circle(overlay, center, radius + 12, (37, 99, 235), 1, cv2.LINE_AA)
        cv2.circle(overlay, center, radius, (255, 255, 255), 2, cv2.LINE_AA)

        for i in range(24):
            angle = math.radians(i * 15 + self._ring_angle)
            if i % 6 == 0:
                tick_len, color, thick = 14, (96, 165, 250), 2
            elif i % 3 == 0:
                tick_len, color, thick = 8, (147, 197, 253), 1
            else:
                tick_len, color, thick = 4, (59, 130, 246), 1
            x1 = int(center[0] + math.cos(angle) * (radius - tick_len // 2))
            y1 = int(center[1] + math.sin(angle) * (radius - tick_len // 2))
            x2 = int(center[0] + math.cos(angle) * (radius + tick_len // 2))
            y2 = int(center[1] + math.sin(angle) * (radius + tick_len // 2))
            cv2.line(overlay, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        self._scan_pos += 0.012 * self._scan_dir
        if self._scan_pos > 1.0:
            self._scan_pos, self._scan_dir = 1.0, -1
        elif self._scan_pos < 0.0:
            self._scan_pos, self._scan_dir = 0.0, 1
        scan_y = int(center[1] - radius + 2 * radius * self._scan_pos)
        scan_alpha_img = overlay.copy()
        cv2.line(scan_alpha_img,
                 (center[0] - radius + 10, scan_y),
                 (center[0] + radius - 10, scan_y),
                 (96, 165, 250), 1, cv2.LINE_AA)
        overlay = cv2.addWeighted(scan_alpha_img, 0.6, overlay, 0.4, 0)
        self._ring_angle = (self._ring_angle + 1.5) % 360.0

        resized = cv2.addWeighted(overlay, 0.65, resized, 0.35, 0)
        qt_img  = QImage(resized.data, rw, rh, rw * 3, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def capture_images(self):
        name, ok = QInputDialog.getText(self, "Nouvel utilisateur", "Nom de l'utilisateur :")
        if not ok or not name:
            return
        num_str, ok2 = QInputDialog.getText(self, "Images par pose", "Nombre d'images par pose (défaut : 40) :")
        try:
            num = int(num_str) if ok2 and num_str else 40
        except ValueError:
            num = 40
        capture_images(name, num_images_per_pose=num)

    def authenticate(self):
        if len(self.known_encodings) == 0:
            self.status_label.setText("⏳  Dataset en cours de chargement...")
            return
        if self.auth_worker is None:
            self.status_label.setText("🔍  Authentification en cours...")
            # Libérer la caméra pour le worker — évite la lenteur
            self.timer.stop()
            self.cap.release()
            self.auth_worker = AuthenticationWorker(
                self.known_encodings, self.known_names, self.known_poses
            )
            self.auth_worker.frame_ready.connect(self._on_auth_frame)
            self.auth_worker.status_update.connect(self._on_auth_status)
            self.auth_worker.progress_update.connect(self._on_auth_progress)
            self.auth_worker.finished.connect(self._on_auth_finished)
            self.auth_worker.start()
        else:
            self.status_label.setText("⚠  Authentification déjà en cours")

    def _on_dataset_loaded(self):
        if len(self.known_names) > 0:
            self.status_label.setText(f"✅  Azure prêt — {len(self.known_names)} utilisateur(s)")
        else:
            self.status_label.setText("⚠  Aucun utilisateur enregistré")

    def _on_auth_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def _on_auth_status(self, txt: str):
        if txt.startswith("TRUST_CODE_READY:"):
            parts = txt.split(":")
            if len(parts) >= 2:
                user_name = parts[1]
                self._show_trust_qr_dialog(user_name)
            return

        self.status_label.setText(txt)
        lower = txt.lower()

        if lower.startswith("ident"):
            parts = txt.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                name = parts[1].strip()
                self.user_name_label.setText(name)
                self.user_avatar.setText("✅")

        if "défi" in lower or "defi" in lower:
            m = re.search(r"(\d+)/(\d+)", txt)
            if m:
                self.challenge_badge.setText(f"Défi {m.group(1)}/{m.group(2)}")
                self.challenge_badge.show()
        elif "✓" in txt or "succès" in lower or "terminée" in lower:
            self.challenge_badge.hide()
            self.status_label.setText("✅  Authentification réussie — Azure Cloud")

    def _show_trust_qr_dialog(self, user_name: str):
        dialog = TrustQRDialog(user_name, parent=self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            # ✅ Code correct → migration autorisée
            self.timer.stop()
            self.cap.release()
            self.migrate_window = MigrateApp()
            self.migrate_window.show()
            self.close()
        else:
            # ❌ Expiré ou fermé → PAS de migration, reprendre la caméra
            self.status_label.setText("⚠  Session expirée — Relancez l'authentification")
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
    def _on_auth_progress(self, val: int):
        self.progress_bar.setValue(val)
        self.prog_pct.setText(f"{val}%")

    def _on_auth_finished(self):
        self.progress_bar.setValue(0)
        self.prog_pct.setText("0%")
        self.challenge_badge.hide()
        try:
            self.auth_worker.quit()
            self.auth_worker.wait(1000)
        except Exception:
            pass

        # La migration est gérée dans _show_trust_qr_dialog()
        # Ici on remet juste l'état par défaut si pas encore migré
        if not self.isHidden():
            self.status_label.setText("Session terminée")
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)

        self.auth_worker = None
    def close_app(self):
        self.timer.stop()
        self.cap.release()
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION CAPTURE (niveau module)
# ══════════════════════════════════════════════════════════════════════════════

def capture_images(user_name, num_images_per_pose=40, output_dir="dataset"):
    dialog = CaptureDialog(user_name)
    poses  = ["frontale", "gauche", "droite", "haut", "bas"]

    capture_thread = CaptureThread(user_name, poses, num_images_per_pose, parent=dialog)
    capture_thread.frame_ready.connect(dialog.set_frame)
    capture_thread.status_update.connect(lambda txt: dialog.status_label.setText(txt))
    capture_thread.progress_update.connect(lambda val: dialog.progress_bar.setValue(val))
    capture_thread.image_captured.connect(lambda pose, path: dialog.update_pose_status(pose))
    capture_thread.start()

    result = dialog.exec()
    if capture_thread.isRunning():
        capture_thread.stop()

    if result == QDialog.DialogCode.Accepted:
        print(f"Capture pour {user_name} terminée.")
    else:
        print("Capture annulée.")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = FaceAuthApp()
    window.show()
    window.showFullScreen()
    sys.exit(app.exec())