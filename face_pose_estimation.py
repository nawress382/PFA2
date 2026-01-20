import cv2
import dlib
import numpy as np
import os
import math
import time
from sklearn.neighbors import NearestNeighbors
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog, QProgressBar, QGraphicsDropShadowEffect
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread

# Import the headless service functions for business logic (API-friendly)
from face_service import (
    estimate_pose as service_estimate_pose,
    get_face_encoding as service_get_face_encoding,
    load_dataset_encodings as service_load_dataset_encodings,
    capture_images_headless as service_capture_images_headless,
    authenticate_sequence as service_authenticate_sequence,
)


class DatasetLoaderThread(QThread):
    """Background thread to load dataset without blocking UI."""
    finished = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
    
    def run(self):
        """Load dataset in background."""
        try:
            print("\n[Dataset Loader] Chargement du dataset en arrière-plan...")
            encodings, names, poses = service_load_dataset_encodings()
            self.parent_app.known_encodings = encodings
            self.parent_app.known_names = names
            self.parent_app.known_poses = poses
            
            print(f"\n=== DIAGNOSTIC POSES APPRISES ===")
            print(f"Utilisateurs: {names}")
            for user, poses_dict in poses.items():
                print(f"\n{user}:")
                for pose_name, (yaw, pitch, roll) in poses_dict.items():
                    print(f"  {pose_name}: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
            print(f"===================================\n")
        except Exception as e:
            print(f"[Dataset Loader] Erreur: {e}")
        finally:
            self.finished.emit()


class FaceAuthApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Authentification Faciale par Rotation de Visage")
        self.setGeometry(100, 100, 800, 600)

        # Load dataset and print diagnostics
        self.known_encodings, self.known_names, self.known_poses = service_load_dataset_encodings()
        print(f"\n=== DIAGNOSTIC POSES APPRISES ===")
        print(f"Utilisateurs: {self.known_names}")
        for user, poses_dict in self.known_poses.items():
            print(f"\n{user}:")
            for pose_name, (yaw, pitch, roll) in poses_dict.items():
                print(f"  {pose_name}: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
        print(f"===================================\n")

        self.setStyleSheet("""
        QWidget { background-color: #0f1222; color: #e6eef8; font-family: 'Segoe UI', Arial; }
        QLabel#titleLabel { font-size: 20px; font-weight: 600; }
        QLabel#statusLabel { color: #bcd3ff; font-size: 14px; }
        QLabel#videoFrame { border-radius: 18px; background-color: #111217; }
        QPushButton { background-color: #3a7afe; color: white; font-size: 14px; padding: 10px 16px; border-radius: 10px; }
        QPushButton#quitButton { background-color: #ff5c5c; }
        QPushButton:hover { background-color: #5790ff; }
        QProgressBar { background-color: #141826; color: #e6eef8; border-radius: 8px; height: 14px; }
        QProgressBar::chunk { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #7bdcff, stop:1 #3a7afe); border-radius: 8px; }
        """)
        
        # Top HUD: title and pose status
        self.title_label = QLabel("Identifié: -")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setStyleSheet("color: #cfefff; font-size: 18px; font-weight: 600;")

        self.pose_status_label = QLabel("")
        self.pose_status_label.setObjectName("poseStatusLabel")
        self.pose_status_label.setStyleSheet("color: #9ee7ff; font-size: 14px; font-weight: 700;")

        # Label pour afficher la vidéo (carré pour anneau circulaire)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(560, 560)
        self.video_label.setObjectName("videoFrame")

        # Status et progression (nécessaires pour l'authentification)
        self.status_label = QLabel("Prêt")
        self.status_label.setObjectName("statusLabel")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        # Boutons
        self.capture_button = QPushButton("Capturer Images")
        self.auth_button = QPushButton("Authentifier")
        self.quit_button = QPushButton("Quitter")

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.auth_button)
        button_layout.addWidget(self.quit_button)

        # Top bar layout
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignLeft)
        top_bar.addStretch(1)
        top_bar.addWidget(self.pose_status_label, alignment=Qt.AlignmentFlag.AlignRight)

        # Icon placeholders row
        icons_layout = QHBoxLayout()
        self.icon_slots = []
        for i in range(4):
            ico = QLabel(self)
            ico.setFixedSize(48, 48)
            ico.setStyleSheet("border-radius: 24px; background-color: rgba(60,80,110,0.25); border: 2px solid rgba(120,200,255,0.08);")
            icons_layout.addStretch(1)
            icons_layout.addWidget(ico)
            self.icon_slots.append(ico)
        icons_layout.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(icons_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
        # Caméra
        self.cap = cv2.VideoCapture(0)
        
        # Animation state for HUD (rotation, scanning line)
        self._ring_angle = 0.0
        self._scan_pos = 0.0
        self._scan_dir = 1

        # Timer pour mise à jour vidéo
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms ~ 33 fps
        
        # Connecter boutons
        self.capture_button.clicked.connect(self.capture_images)
        self.auth_button.clicked.connect(self.authenticate)
        self.quit_button.clicked.connect(self.close_app)

        # Authentication worker placeholder
        self.auth_worker = None
        
        # Load dataset in background thread (non-blocking)
        self.dataset_loader = DatasetLoaderThread(self)
        self.dataset_loader.finished.connect(self._on_dataset_loaded)
        self.dataset_loader.start()
    
    

    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert to RGB and prepare a square crop centered on the face/camera
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            size = min(h, w)
            cx, cy = w // 2, h // 2
            half = size // 2
            crop = frame[cy - half:cy + half, cx - half:cx + half].copy()

            # Resize crop to the video_label size
            label_w = self.video_label.width()
            label_h = self.video_label.height()
            if crop.shape[0] != label_h or crop.shape[1] != label_w:
                crop = cv2.resize(crop, (label_w, label_h), interpolation=cv2.INTER_LINEAR)

            rh, rw, _ = crop.shape

            # Draw neon rotating rings and tick marks
            overlay = crop.copy()
            center = (rw // 2, rh // 2)
            radius = int(min(rw, rh) * 0.38)

            # outer glow
            cv2.circle(overlay, center, radius + 8, (10, 150, 170), thickness=10)
            # inner ring
            cv2.circle(overlay, center, radius, (100, 220, 200), thickness=3)

            # rotating ticks
            num_ticks = 16
            for i in range(num_ticks):
                angle = math.radians(i * (360.0 / num_ticks) + self._ring_angle)
                x1 = int(center[0] + math.cos(angle) * (radius - 6))
                y1 = int(center[1] + math.sin(angle) * (radius - 6))
                x2 = int(center[0] + math.cos(angle) * (radius + 6))
                y2 = int(center[1] + math.sin(angle) * (radius + 6))
                # vary alpha/brightness for visual effect
                color = (100, 220, 200)
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # scanning line animation (vertical sweep inside circle)
            # update animation state
            self._ring_angle = (self._ring_angle + 2.0) % 360.0
            self._scan_pos += 0.015 * self._scan_dir
            if self._scan_pos > 1.0:
                self._scan_pos = 1.0
                self._scan_dir = -1
            elif self._scan_pos < 0.0:
                self._scan_pos = 0.0
                self._scan_dir = 1

            scan_y = int(center[1] - radius + 2 * radius * self._scan_pos)
            cv2.line(overlay, (center[0] - radius + 4, scan_y), (center[0] + radius - 4, scan_y), (60, 200, 240), 2)

            # radial progress arc based on progress_bar value
            try:
                progress = int(self.progress_bar.value())
            except Exception:
                progress = 0
            if progress > 0:
                # draw an arc from -90 degrees to -90 + progress*3.6
                start_angle = -90
                end_angle = -90 + int(progress * 3.6)
                # cv2.ellipse expects angle in degrees and uses axes
                cv2.ellipse(overlay, center, (radius - 18, radius - 18), 0, start_angle, end_angle, (6, 182, 212), 6)

            # blend overlay with crop
            crop = cv2.addWeighted(overlay, 0.7, crop, 0.3, 0)

            # HUD text
            txt = "Alignez votre visage dans le cercle"
            (txt_w, txt_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(crop, txt, (max(10, rw//2 - txt_w//2), int(rh*0.92)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 230, 240), 2)

            # Convert to QImage and display
            bytes_per_line = 3 * rw
            qt_image = QImage(crop.data, rw, rh, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pix)
    
    def capture_images(self):
        # Demander le nom de l'utilisateur et le nombre d'images par pose via la GUI
        name, ok = QInputDialog.getText(self, "Nom utilisateur", "Entrez le nom de l'utilisateur:")
        if not ok or not name:
            print("Capture annulée: aucun nom fourni.")
            return

        num_str, ok2 = QInputDialog.getText(self, "Nombre images", "Entrez le nombre d'images par pose (par défaut 40):")
        try:
            num = int(num_str) if ok2 and num_str else 40
        except ValueError:
            num = 40

        capture_images(name, num_images_per_pose=num)  # Appelle la fonction existante
    def authenticate(self):
        # Charger le dataset (ou utiliser les données chargées en arrière-plan)
        if len(self.known_encodings) == 0:
            self.status_label.setText("Dataset en cours de chargement... Réessayez dans quelques secondes.")
            return

        # Démarrer le worker d'authentification qui mettra à jour l'UI
        if self.auth_worker is None:
            self.status_label.setText("Démarrage de l'authentification...")
            self.auth_worker = AuthenticationWorker(self.known_encodings, self.known_names, self.known_poses)
            self.auth_worker.frame_ready.connect(self._on_auth_frame)
            self.auth_worker.status_update.connect(self._on_auth_status)
            self.auth_worker.progress_update.connect(self._on_auth_progress)
            self.auth_worker.finished.connect(self._on_auth_finished)
            self.auth_worker.start()
        else:
            # Si déjà en cours, ignorer ou arrêter
            self.status_label.setText("Authentification déjà en cours...")

    def _on_dataset_loaded(self):
        """Called when dataset is loaded in background."""
        if len(self.known_names) > 0:
            self.status_label.setText(f"✓ Dataset chargé ({len(self.known_names)} utilisateurs)")
        else:
            self.status_label.setText("⚠ Aucun utilisateur dans le dataset")

    def _on_auth_frame(self, qimg):
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def _on_auth_status(self, txt: str):
        # Update main status and HUD labels
        self.status_label.setText(txt)
        lower = txt.lower()

        # If identification message, update title
        if lower.startswith("ident"):
            parts = txt.split(":", 1)
            if len(parts) > 1:
                name = parts[1].strip()
                if name:
                    self.title_label.setText(f"Identifié: {name}")

        # If challenge message, try to extract pose name and display
        if "défi" in lower or "defi" in lower:
            # look for single-quoted pose like 'droite'
            import re
            m = re.search(r"pose\s*'([^']+)'", txt, flags=re.IGNORECASE)
            if m:
                pose_name = m.group(1)
                self.pose_status_label.setText(f"Demande: {pose_name}")
            else:
                # fallback to show short text
                self.pose_status_label.setText(txt if len(txt) < 30 else txt[:30] + "...")
        elif "pose correcte" in lower or "pose ok" in lower or "authentification réussie" in lower:
            self.pose_status_label.setText("Pose OK")
        elif "inconnu" in lower or "erreur" in lower:
            # clear or show warning
            self.pose_status_label.setText("")

    def _on_auth_progress(self, val: int):
        self.progress_bar.setValue(val)

    def _on_auth_finished(self):
        self.status_label.setText("Authentification terminée")
        self.progress_bar.setValue(0)
        # cleanup
        try:
            self.auth_worker.quit()
            self.auth_worker.wait(1000)
        except Exception:
            pass
        self.auth_worker = None

    
    def close_app(self):
        self.cap.release()
        self.close()

# --- Constantes (assure-toi que les chemins sont corrects) ---
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
DATASET_DIR = "dataset"
FACE_DETECTION_THRESHOLD = 0.6  # Seuil de détection pour dlib (pour l'identification)

# Paramètres de capture automatique / stabilité
MIN_TIME_BETWEEN_SAVES = 0.7    # secondes minimum entre deux captures
WAIT_BETWEEN_POSES = 5         # secondes d'attente avant et après chaque pose
STABLE_FRAMES_REQUIRED = 3     # nombre d'images consécutives stables nécessaires
STABILITY_THRESH = 20          # pixels (distance maximale entre centres pour considérer stable)
TOTAL_CHALLENGES = 2          # nombre de défis successifs requis pour authentification
CHALLENGE_TIMEOUT = 12        # secondes max pour réussir chaque défi (sinon recommencer)

# --- Initialisation de dlib ---
detector = dlib.get_frontal_face_detector()

if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"Erreur: Le modèle shape_predictor est introuvable à {SHAPE_PREDICTOR_PATH}")
    print("Veuillez télécharger shape_predictor_68_face_landmarks.dat.bz2 et le décompresser.")
    exit()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
    print(f"Erreur: Le modèle de reconnaissance faciale est introuvable à {FACE_RECOGNITION_MODEL_PATH}")
    print("Veuillez télécharger dlib_face_recognition_resnet_model_v1.dat.bz2 et le décompresser.")
    exit()
face_recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# --- Paramètres de la caméra pour l'estimation de pose ---
CAMERA_MATRIX = np.array([
    [0.0, 0, 0],
    [0, 0.0, 0],
    [0, 0, 1]
], dtype="double")

DIST_COEFFS = np.zeros((4, 1))

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (30)
    (0.0, -330.0, -65.0),        # Chin (8)
    (-225.0, 170.0, -135.0),     # Left eye left corner (36)
    (225.0, 170.0, -135.0),      # Right eye right corner (45)
    (-150.0, -150.0, -125.0),    # Left mouth corner (48)
    (150.0, -150.0, -125.0)      # Right mouth corner (54)
], dtype="double")

# Paramètres pour le rendu du maillage 3D : décalage du centre du maillage
# par rapport au point modèle (nose tip). Cela permet de centrer l'ellipsoïde
# sur la tête/visage et d'éviter qu'il soit collé à la pointe du nez.
# Valeurs en mm dans le même repère que MODEL_POINTS: (x, y, z)
# - x: décalage gauche/droite (positif vers la droite)
# - y: décalage haut/bas (positif vers le haut)
# - z: décalage avant/arriére (négatif éloigne du plan de la caméra dans ce repère)
MESH_CENTER_OFFSET = (0.0, 80.0, -140.0)

# --- Fonctions auxiliaires ---

def get_face_landmarks(image_rgb, face_rect):
    landmarks = predictor(image_rgb, face_rect)
    return landmarks

def get_face_encoding(image_rgb, face_rect):
    landmarks = predictor(image_rgb, face_rect)
    return np.array(face_recognizer.compute_face_descriptor(image_rgb, landmarks))

def estimate_pose(landmarks, img_w, img_h):
    global CAMERA_MATRIX
    
    FOCAL_LENGTH_ESTIMATED = img_w 
    
    CAMERA_MATRIX[0, 0] = FOCAL_LENGTH_ESTIMATED
    CAMERA_MATRIX[1, 1] = FOCAL_LENGTH_ESTIMATED
    CAMERA_MATRIX[0, 2] = img_w / 2
    CAMERA_MATRIX[1, 2] = img_h / 2

    image_points_selected = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")

    (success, rvec, tvec) = cv2.solvePnP(MODEL_POINTS, image_points_selected, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return 0, 0, 0, None, None

    rot_matrix, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rot_matrix, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]

    yaw_deg = -yaw[0]
    pitch_deg = pitch[0]
    roll_deg = -roll[0]

    return yaw_deg, pitch_deg, roll_deg, rvec, tvec

def draw_pose_info(frame, yaw, pitch, roll):
    # Pour une UI plus propre, ne pas afficher yaw/pitch/roll directement sur la vidéo.
    # Si nécessaire, remplacer par des icônes ou indicateurs plus discrets.
    return


def draw_3d_mesh(frame, rvec, tvec, landmarks=None, color=(110, 160, 255), lat_steps=8, lon_steps=16):
    """Dessine un maillage 3D (ellipsoïde approximatif) projeté sur l'image en utilisant rvec/tvec.
    L'ellipsoïde est centrée autour du point modèle (nose tip) dans l'espace du modèle.
    Ceci donne un effet de 'maillage' 3D sur la face pendant l'authentification.
    """
    if rvec is None or tvec is None:
        return

    # Paramètres de l'ellipsoïde (en mm, dans le repère du modèle)
    # Ces valeurs représentent approximativement la taille du maillage autour
    # de la tête ; on peut les ajuster si nécessaire.
    rx, ry, rz = 120.0, 150.0, 110.0

    verts = []
    for i in range(1, lat_steps + 1):
        lat = (i / (lat_steps + 1.0) - 0.5) * math.pi  # -pi/2..pi/2
        for j in range(lon_steps):
            lon = (j / float(lon_steps)) * 2.0 * math.pi - math.pi
            x = rx * math.cos(lat) * math.sin(lon)
            y = ry * math.sin(lat)
            z = rz * math.cos(lat) * math.cos(lon)
            verts.append((x, y, z))

    if not verts:
        return

    pts3d = np.array(verts, dtype='double')

    # Appliquer un décalage de centre pour que l'ellipsoïde soit centré
    # sur la tête (et non sur la pointe du nez). Le décalage est en mm
    # dans le repère du modèle et peut être ajusté via MESH_CENTER_OFFSET.
    try:
        offset = np.array(MESH_CENTER_OFFSET, dtype='double')
        pts3d = pts3d + offset.reshape((1, 3))
    except Exception:
        # En cas de problème, continuer sans offset
        pass
    # Project 3D points to 2D image points
    try:
        imgpts, _ = cv2.projectPoints(pts3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    except Exception:
        return

    imgpts = imgpts.reshape(-1, 2)

    # If landmarks are provided, compute an anatomical desired center in 2D
    # (midpoint between eyes and chin) and translate the projected points so
    # the mesh visually centers on that point rather than nose tip.
    try:
        if landmarks is not None:
            left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y], dtype=float)
            right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y], dtype=float)
            chin = np.array([landmarks.part(8).x, landmarks.part(8).y], dtype=float)
            eyes_mid = (left_eye + right_eye) / 2.0
            desired_center = (eyes_mid + chin) / 2.0

            centroid = np.mean(imgpts, axis=0)
            delta = desired_center - centroid
            imgpts = imgpts + delta.reshape((1, 2))
    except Exception:
        pass

    # draw horizontal (longitude) lines
    for i in range(lat_steps):
        for j in range(lon_steps):
            idx = i * lon_steps + j
            p1 = tuple(int(v) for v in imgpts[idx])
            p2 = tuple(int(v) for v in imgpts[i * lon_steps + ((j + 1) % lon_steps)])
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

    # draw vertical (latitude) lines
    for j in range(lon_steps):
        for i in range(lat_steps - 1):
            idx = i * lon_steps + j
            p1 = tuple(int(v) for v in imgpts[idx])
            p2 = tuple(int(v) for v in imgpts[(i + 1) * lon_steps + j])
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

    # Dessiner un petit marqueur (cercles) au centre projeté de l'ellipsoïde
    # pour aider au réglage visuel. On projette le barycentre des points 3D
    # (après application de l'offset) pour obtenir la position 2D à dessiner.
    try:
        center3d = np.mean(pts3d, axis=0).reshape((1, 3))
        center2d, _ = cv2.projectPoints(center3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
        c = tuple(int(v) for v in center2d.reshape(2,))
        # cercle extérieur puis intérieur pour meilleure visibilité
        cv2.circle(frame, c, 8, (0, 180, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
    except Exception:
        pass


# --- Fonctions de capture d'images ---

def capture_images(user_name, num_images_per_pose=40, output_dir="dataset"):
    user_output_dir = os.path.join(output_dir, user_name)
    os.makedirs(user_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra pour la capture d'images.")
        return

    print(f"Préparation à la capture pour {user_name}. Appuyez sur 'q' pour quitter.")
    print("Veuillez regarder de face, puis tourner légèrement la tête à gauche, à droite, en haut, en bas pour chaque série.")
    
    captured_count = 0
    poses = ["frontale", "gauche", "droite", "haut", "bas"]
    images_to_capture_total = num_images_per_pose * len(poses)
    
    print(f"Total d'images à capturer : {images_to_capture_total}")

    for pose_index, pose in enumerate(poses):
        print(f"\nPréparez-vous pour la pose : {pose}. Capture de {num_images_per_pose} images.")
        print(f"Attente {WAIT_BETWEEN_POSES} secondes avant le début de la capture pour cette pose...")
        time.sleep(WAIT_BETWEEN_POSES)

        current_pose_count = 0
        # Pour la capture automatique, on garde le temps de la dernière sauvegaFrde
        last_save_time = 0
        # stockage des derniers centres pour vérifier la stabilité
        last_centers = []
        while current_pose_count < num_images_per_pose:
            ret, frame = cap.read()

            if not ret:
                print("Erreur : Impossible de lire l'image de la caméra.")
                break

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) == 1:
                face_rect = faces[0]
                x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(display_frame, f"Pose {pose}: {current_pose_count+1}/{num_images_per_pose}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Capture automatique active... (stabilité requise)", 
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

                # Vérifier la stabilité : utiliser le centre du rectangle
                cx = x + w / 2
                cy = y + h / 2
                last_centers.append((cx, cy))
                # ne garder que les derniers STABLE_FRAMES_REQUIRED centres
                if len(last_centers) > STABLE_FRAMES_REQUIRED:
                    last_centers.pop(0)

                stable = False
                if len(last_centers) == STABLE_FRAMES_REQUIRED:
                    # calculer la distance max entre centres
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

                # Sauvegarde automatique si stable et assez de temps est passé depuis la dernière capture
                now = time.time()
                if stable and (now - last_save_time >= MIN_TIME_BETWEEN_SAVES) and current_pose_count < num_images_per_pose:
                    img_path = os.path.join(user_output_dir, f"{user_name}_{pose}_{current_pose_count:03d}.jpg")
                    cv2.imwrite(img_path, frame)
                    print(f"Image sauvegardée automatiquement (stable) : {img_path}")
                    current_pose_count += 1
                    captured_count += 1
                    last_save_time = now

            else:
                # reset stability tracking si visage perdu
                last_centers = []
                cv2.putText(display_frame, "Placez votre visage au centre", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Capture Images', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            # L'utilisateur peut toujours appuyer sur 'q' pour annuler la capture
            if key == ord('q'):
                print("Capture annulée.")
                cap.release()
                cv2.destroyAllWindows()
                return

        if current_pose_count == num_images_per_pose:
            print(f"Capture de la pose '{pose}' terminée.")
            print("Pause 5 secondes avant la pose suivante...")
            time.sleep(5)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture d'images pour {user_name} terminée. Total : {captured_count} images.")

# --- Fonctions de chargement du dataset ---

def load_dataset_encodings(dataset_dir=DATASET_DIR):
    known_encodings = []
    known_names = []
    known_poses = {}
    
    if not os.path.exists(dataset_dir):
        print(f"Avertissement: Le répertoire du dataset '{dataset_dir}' n'existe pas.")
        return np.array([]), [], {}

    for user_name in os.listdir(dataset_dir):
        user_path = os.path.join(dataset_dir, user_name)
        if os.path.isdir(user_path):
            print(f"DEBUG: Traitement de l'utilisateur: {user_name}")
            current_user_encodings = []
            current_user_poses_data = {
                "frontale": [], "gauche": [], "droite": [], "haut": [], "bas": []
            }

            for img_name in os.listdir(user_path):
                if img_name.endswith(".jpg") or img_name.endswith(".png"):
                    img_path = os.path.join(user_path, img_name)
                    print(f"DEBUG: Traitement de l'image: {img_name}")

                    image_bgr = cv2.imread(img_path)
                    if image_bgr is None:
                        print(f"Avertissement: Impossible de lire l'image {img_path}")
                        continue
                    
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) 
                    
                    # On utilise l'upsampling de dlib ici pour être plus robuste sur les images du dataset
                    faces = detector(gray, 1) # Upsample une fois

                    if len(faces) == 1:
                        face = faces[0]
                        
                        try:
                            encoding = get_face_encoding(image_rgb, face)
                            current_user_encodings.append(encoding)
                        except Exception as e:
                            print(f"ERREUR DEBUG: Impossible d'encoder le visage pour {img_name}: {e}")
                            continue

                        try:
                            landmarks = predictor(image_rgb, face)
                            img_h, img_w = image_rgb.shape[:2]
                            yaw, pitch, roll, _, _ = estimate_pose(landmarks, img_w, img_h)
                            print(f"DEBUG: Pose estimée pour {img_name}: Yaw={yaw:.1f}, Pitch={pitch:.1f}, Roll={roll:.1f}")
                        except Exception as e:
                            print(f"ERREUR DEBUG: Impossible d'estimer la pose pour {img_name}: {e}")
                            continue

                        found_pose_key = False
                        for pose_key in current_user_poses_data.keys():
                            if pose_key in img_name.lower():
                                current_user_poses_data[pose_key].append((yaw, pitch, roll))
                                print(f"DEBUG: Image {img_name} associée à la pose: {pose_key}")
                                found_pose_key = True
                                break
                        if not found_pose_key:
                            print(f"DEBUG: Image {img_name} n'a pas pu être associée à une pose connue.")
                    else:
                        print(f"Avertissement: 0 ou >1 visage détecté dans {img_name}. Ignoré pour encodage/pose.")
            
            if current_user_encodings:
                avg_encoding = np.mean(current_user_encodings, axis=0)
                known_encodings.append(avg_encoding)
                known_names.append(user_name)

                avg_poses_for_user = {}
                for pose_key, ypr_list in current_user_poses_data.items():
                    if ypr_list:
                        avg_poses_for_user[pose_key] = np.mean(ypr_list, axis=0)
                        print(f"DEBUG: Pose '{pose_key}' pour {user_name} contient {len(ypr_list)} enregistrements. Moyenne: {avg_poses_for_user[pose_key]}")
                known_poses[user_name] = avg_poses_for_user
            else:
                print(f"DEBUG: Aucun encodage facial traité pour {user_name}. Les poses ne seront pas enregistrées.")

    print(f"Dataset chargé. {len(known_names)} utilisateurs enregistrés.")
    print(f"DEBUG FINAL known_poses: {known_poses}")
    return np.array(known_encodings), known_names, known_poses

# --- Fonction principale d'authentification ---

def recognize_and_authenticate_with_pose(known_encodings, known_names, known_poses):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra pour l'authentification.")
        return

    if len(known_encodings) > 0:
        face_recognizer_knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        face_recognizer_knn.fit(known_encodings)
    else:
        print("Aucun encodage facial enregistré. L'authentification ne peut pas démarrer.")
        cap.release()
        cv2.destroyAllWindows()
        return

    authenticated_user = None
    authentication_stage = "IDLE" # IDLE, POSE_CHALLENGE, AUTHENTICATED
    required_pose = None
    pose_start_time = 0
    POSE_DURATION = 3 # Secondes pour maintenir la pose

    TOLERANCE_YAW = 15
    TOLERANCE_PITCH = 15
    TOLERANCE_ROLL = 15

    print("\n--- Démarrage de l'authentification faciale par pose ---")
    print("Placez votre visage devant la caméra. Une fois identifié, suivez les instructions de pose.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire l'image de la caméra pendant l'authentification.")
            break

        # Tenter d'augmenter fx_scale pour une meilleure détection
        fx_scale = 0.75 # Passé de 0.5 à 0.75
        small_frame = cv2.resize(frame, (0, 0), fx=fx_scale, fy=fx_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) 
        gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Utiliser un upsampling (1) pour dlib sur la petite frame pour trouver des visages plus petits
        faces = detector(gray_small_frame, 1) 
        
        display_frame = frame.copy() 

        if len(faces) > 0:
            face = faces[0] 
            
            # Ajuster les coordonnées du visage détecté sur la petite frame à la taille originale de la frame
            face_rect_original_coords = dlib.rectangle(
                int(face.left() / fx_scale), int(face.top() / fx_scale), 
                int(face.right() / fx_scale), int(face.bottom() / fx_scale)
            )
            
            x, y, w, h = face_rect_original_coords.left(), face_rect_original_coords.top(), face_rect_original_coords.width(), face_rect_original_coords.height()
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Obtenir les landmarks sur la frame originale pour une meilleure précision de pose
            # (Note: predictor peut prendre la frame originale et la face_rect_original_coords)
            landmarks_original_coords = predictor(frame, face_rect_original_coords)
            
            img_h, img_w = frame.shape[:2]
            current_yaw, current_pitch, current_roll, _, _ = estimate_pose(landmarks_original_coords, img_w, img_h)
            draw_pose_info(display_frame, current_yaw, current_pitch, current_roll)

            if authentication_stage == "IDLE":
                # Utiliser la petite frame RGB pour l'encodage pour la performance,
                # mais la face_rect doit être la version redimensionnée pour la small_frame
                # Correction: get_face_encoding prend l'image et la face_rect correspondante.
                # Ici, nous utilisons rgb_small_frame et la 'face' détectée sur small_frame.
                try:
                    face_encoding = get_face_encoding(rgb_small_frame, face) 
                    
                    distances, indices = face_recognizer_knn.kneighbors([face_encoding])
                    min_distance = distances[0][0]
                    identified_index = indices[0][0]

                    if min_distance < FACE_DETECTION_THRESHOLD:
                        identified_user = known_names[identified_index]
                        cv2.putText(display_frame, f"Identifie: {identified_user}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        authenticated_user = identified_user
                        print(f"Utilisateur identifié: {authenticated_user}. Préparation du défi de pose.")
                        
                        # S'assurer que known_poses[authenticated_user] existe et n'est pas vide
                        user_poses_data = known_poses.get(authenticated_user)
                        if user_poses_data and len(user_poses_data) > 0:
                            available_poses = list(user_poses_data.keys())
                            required_pose = np.random.choice(available_poses)
                            authentication_stage = "POSE_CHALLENGE"
                            print(f"Défi de pose pour {authenticated_user}: Faites la pose '{required_pose}'.")
                            pose_start_time = time.time()
                        else:
                            print(f"Avertissement: Aucune pose enregistrée pour {authenticated_user}. Authentification impossible. Retour à IDLE.")
                            cv2.putText(display_frame, "Pas de poses enregistrees", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            authentication_stage = "IDLE" 
                    else:
                        cv2.putText(display_frame, "Inconnu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Erreur lors de l'encodage ou de la reconnaissance: {e}")
                    cv2.putText(display_frame, "Erreur Reconnaissance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    authentication_stage = "IDLE"

            elif authentication_stage == "POSE_CHALLENGE":
                cv2.putText(display_frame, f"Utilisateur: {authenticated_user}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Faites la pose: {required_pose}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if required_pose and authenticated_user in known_poses:
                    target_ypr = known_poses[authenticated_user].get(required_pose)
                    if target_ypr is not None:
                        yaw_match = abs(current_yaw - target_ypr[0]) < TOLERANCE_YAW
                        pitch_match = abs(current_pitch - target_ypr[1]) < TOLERANCE_PITCH
                        roll_match = abs(current_roll - target_ypr[2]) < TOLERANCE_ROLL
                        
                        if yaw_match and pitch_match and roll_match:
                            cv2.putText(display_frame, "Pose CORRESPOND", (img_w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            elapsed_time = time.time() - pose_start_time
                            remaining_time = max(0, POSE_DURATION - elapsed_time)
                            cv2.putText(display_frame, f"Maintenez: {remaining_time:.1f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            if elapsed_time >= POSE_DURATION:
                                authentication_stage = "AUTHENTICATED"
                                print(f"Authentification réussie pour {authenticated_user} !")
                                cv2.putText(display_frame, "AUTHENTIFIE !", (img_w // 2 - 100, img_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                                cv2.imshow('Authentification', display_frame)
                                cv2.waitKey(3000)
                                authentication_stage = "IDLE" 
                                authenticated_user = None 
                                required_pose = None
                                print("Authentification terminee, retour a l'etat IDLE.")
                        else:
                            cv2.putText(display_frame, "Pose INCORRECTE", (img_w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            # Réinitialiser le timer si la pose n'est plus correcte
                            pose_start_time = time.time() if pose_start_time == 0 else pose_start_time
                            cv2.putText(display_frame, "Re-positionnez-vous", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(display_frame, f"Erreur: Pose '{required_pose}' introuvable pour {authenticated_user}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        authentication_stage = "IDLE" # Revenir à IDLE si la pose requise n'est pas trouvée

            elif authentication_stage == "AUTHENTICATED":
                # Cet état est géré par la pause après l'authentification réussie
                pass # Rien à faire ici, le code ci-dessus gère déjà la transition

        else: # Aucun visage détecté
            cv2.putText(display_frame, "Aucun visage detecte", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Réinitialiser tous les états si le visage disparaît
            if authentication_stage != "IDLE":
                print("Visage perdu, retour a l'etat IDLE.")
            authentication_stage = "IDLE" 
            authenticated_user = None
            required_pose = None

        cv2.imshow('Authentification', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Authentification terminee.")


class AuthenticationWorker(QThread):
    """Worker QThread that runs the authentication loop and emits frames and status updates."""
    frame_ready = pyqtSignal(QImage)
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)

    def __init__(self, known_encodings, known_names, known_poses, parent=None):
        super().__init__(parent)
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.known_poses = known_poses
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_update.emit("Erreur : Impossible d'ouvrir la caméra.")
            return

        if len(self.known_encodings) > 0:
            face_recognizer_knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            face_recognizer_knn.fit(self.known_encodings)
        else:
            self.status_update.emit("Aucun encodage facial enregistré.")
            cap.release()
            return

        POSE_DURATION = 3
        TOLERANCE_YAW = 15
        TOLERANCE_PITCH = 15
        TOLERANCE_ROLL = 15

        authenticated_user = None
        authentication_succeeded = False

        try:
            while self._running and not authentication_succeeded:
                # read frame
                ret, frame = cap.read()
                if not ret:
                    self.status_update.emit("Erreur lecture caméra")
                    break

                small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_small, 1)

                display = frame.copy()

                if authenticated_user is None:
                    # try to identify
                    if len(faces) > 0:
                        try:
                            enc = get_face_encoding(rgb_small, faces[0])
                            distances, indices = face_recognizer_knn.kneighbors([enc])
                            min_distance = distances[0][0]
                            idx = indices[0][0]
                            if min_distance < FACE_DETECTION_THRESHOLD:
                                authenticated_user = self.known_names[idx]
                                self.status_update.emit(f"Identifié: {authenticated_user}")
                                # ensure poses exist for user
                                user_poses = self.known_poses.get(authenticated_user)
                                if not user_poses or len(user_poses) == 0:
                                    self.status_update.emit("Aucune pose enregistrée pour cet utilisateur")
                                    authenticated_user = None
                            else:
                                self.status_update.emit("Inconnu - en attente...")
                        except Exception as e:
                            self.status_update.emit(f"Erreur reconnaissance: {e}")
                    else:
                        self.status_update.emit("Aucun visage détecté")

                else:
                    # run sequence of challenges
                    user_poses = self.known_poses.get(authenticated_user, {})
                    poses_list = list(user_poses.keys())
                    if not poses_list:
                        self.status_update.emit("Aucune pose disponible pour l'utilisateur")
                        authenticated_user = None
                        continue

                    challenge_index = 0
                    while challenge_index < TOTAL_CHALLENGES and self._running:
                        required_pose = np.random.choice(poses_list)
                        self.status_update.emit(f"Défi {challenge_index+1}/{TOTAL_CHALLENGES} : Faites la pose '{required_pose}'")
                        challenge_start = time.time()
                        hold_start = None
                        succeeded = False

                        while (time.time() - challenge_start) <= CHALLENGE_TIMEOUT and self._running:
                            ret2, frame2 = cap.read()
                            if not ret2:
                                break
                            small2 = cv2.resize(frame2, (0, 0), fx=0.75, fy=0.75)
                            gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
                            faces2 = detector(gray2, 1)
                            display2 = frame2.copy()

                            if len(faces2) > 0:
                                face2 = faces2[0]
                                face_rect_original = dlib.rectangle(
                                    int(face2.left() / 0.75), int(face2.top() / 0.75),
                                    int(face2.right() / 0.75), int(face2.bottom() / 0.75)
                                )

                                landmarks = predictor(frame2, face_rect_original)
                                yaw, pitch, roll, rvec, tvec = estimate_pose(landmarks, frame2.shape[1], frame2.shape[0])

                                target = user_poses.get(required_pose)
                                if target is None:
                                    break
                                
                                # Adaptive tolerances per pose (gauche/droite need wider yaw)
                                pose_tol = {
                                    "frontale": (TOLERANCE_YAW, TOLERANCE_PITCH, TOLERANCE_ROLL),
                                    "gauche": (25, TOLERANCE_PITCH, TOLERANCE_ROLL),
                                    "droite": (25, TOLERANCE_PITCH, TOLERANCE_ROLL),
                                    "haut": (TOLERANCE_YAW, 20, TOLERANCE_ROLL),
                                    "bas": (TOLERANCE_YAW, 20, TOLERANCE_ROLL),
                                }
                                tol_yaw, tol_pitch, tol_roll = pose_tol.get(required_pose, (TOLERANCE_YAW, TOLERANCE_PITCH, TOLERANCE_ROLL))
                                
                                yaw_err = abs(yaw - target[0])
                                pitch_err = abs(pitch - target[1])
                                roll_err = abs(roll - target[2])
                                
                                yaw_match = yaw_err < tol_yaw
                                pitch_match = pitch_err < tol_pitch
                                roll_match = roll_err < tol_roll
                                
                                # DEBUG: show mismatches
                                diagnostic = f"{required_pose}| Y:{yaw:.1f}→{target[0]:.1f}(Δ{yaw_err:.1f}/{tol_yaw}) P:{pitch:.1f}→{target[1]:.1f}(Δ{pitch_err:.1f}/{tol_pitch}) R:{roll:.1f}→{target[2]:.1f}(Δ{roll_err:.1f}/{tol_roll})"
                                self.status_update.emit(diagnostic)

                                if yaw_match and pitch_match and roll_match:
                                    if hold_start is None:
                                        hold_start = time.time()
                                    elapsed_hold = time.time() - hold_start
                                    percent = int(min(100, (elapsed_hold / POSE_DURATION) * 100))
                                    self.progress_update.emit(percent)
                                    if elapsed_hold >= POSE_DURATION:
                                        succeeded = True
                                        self.status_update.emit(f"Défi {challenge_index+1} réussi")
                                        time.sleep(0.6)
                                        break
                                else:
                                    hold_start = None
                                    self.progress_update.emit(0)
                            else:
                                hold_start = None
                                self.progress_update.emit(0)

                            # emit current frame
                            rgb_display = cv2.cvtColor(display2, cv2.COLOR_BGR2RGB)
                            h, w, ch = rgb_display.shape
                            bytes_per_line = ch * w
                            qimg = QImage(rgb_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                            self.frame_ready.emit(qimg)
                            time.sleep(0.02)

                        if not succeeded:
                            # challenge failed -> restart entire authentication
                            self.status_update.emit("Défi échoué : recommencer l'authentification...")
                            self.progress_update.emit(0)
                            authenticated_user = None
                            time.sleep(1.0)
                            break
                        else:
                            challenge_index += 1

                    # if all challenges succeeded
                    if challenge_index >= TOTAL_CHALLENGES:
                        self.status_update.emit(f"Authentification réussie pour {authenticated_user} !")
                        self.progress_update.emit(100)
                        time.sleep(1.0)
                        authentication_succeeded = True
                        break

                # emit a frame if none emitted in inner loops
                try:
                    rgb_display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_display.shape
                    bytes_per_line = ch * w
                    qimg = QImage(rgb_display.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.frame_ready.emit(qimg)
                except Exception:
                    pass

                time.sleep(0.02)

        finally:
            try:
                cap.release()
            except Exception:
                pass
            self.progress_update.emit(0)
            self.status_update.emit("Finished")
            self.frame_ready.emit(QImage())

# --- Bloc principal d'exécution ---
if __name__ == "__main__":




    app = QApplication(sys.argv)
    window = FaceAuthApp()
    window.show()
    sys.exit(app.exec())
    # Pour CAPTURER de nouvelles images :
    # 1. Décommenter la ligne `capture_images()` ci-dessous.
    # 2. Remplacer "NomUtilisateur" par le nom souhaité.
    # 3. Commenter toutes les lignes de la partie "AUTHENTIFICATION".
    # 4. Exécuter le script.
    # 5. Une fois les images capturées, RECOMMENTER `capture_images()` et DÉCOMMENTER la partie AUTHENTIFICATION.

    # Exemple de capture pour un nouvel utilisateur :
    capture_images("Mariem", num_images_per_pose=40) 
    
    # --- PARTIE AUTHENTIFICATION (DÉCOMMENTÉE PAR DÉFAUT) ---
    print("Chargement du dataset et démarrage de l'authentification...")
    known_encodings, known_names, known_poses = load_dataset_encodings()
    
    print(f"\nKnown Names (utilisateurs enregistrés): {known_names}")
    for name, poses in known_poses.items():
        print(f"  {name} Poses enregistrées: {list(poses.keys())}")

    if len(known_encodings) > 0:
        recognize_and_authenticate_with_pose(known_encodings, known_names, known_poses)
    else:
        print("\nATTENTION: Aucun utilisateur enregistré dans le dataset.")
        print("Veuillez d'abord capturer des images pour au moins un utilisateur en décommentant la ligne 'capture_images(\"NomUtilisateur\")'.")