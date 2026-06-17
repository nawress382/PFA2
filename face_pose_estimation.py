import cv2
import dlib
import numpy as np
import os
import math
import time
from sklearn.neighbors import NearestNeighbors
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QInputDialog, QProgressBar, QGraphicsDropShadowEffect, QDialog, QFrame, QLineEdit
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread
import boto3
from security_utils import encrypt_aes_512, upload_to_s3
import uuid  # <--- CELUI QUI MANQUAIT
from security_utils import generate_and_upload_qr
import webbrowser




# Import the headless service functions for business logic (API-friendly)
from face_service import (
    estimate_pose as service_estimate_pose,
    get_face_encoding as service_get_face_encoding,
    load_dataset_encodings as service_load_dataset_encodings,
    capture_images_headless as service_capture_images_headless,
    authenticate_sequence as service_authenticate_sequence,
)
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('FaceAuthTasks')
MON_BUCKET = "pfe-face-auth-storage-votre-nom" # Mets ton vrai nom de bucket S3

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

class CaptureThread(QThread):

    """Fil d'exécution pour la capture vidéo avec synchronisation Cloud DynamoDB."""
    
    frame_ready = pyqtSignal(QImage)
    image_captured = pyqtSignal(str, str)  # pose, image_path
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, user_name, poses_list, num_per_pose=40, parent=None):
        super().__init__(parent)
        self.user_name = user_name
        self.poses_list = poses_list
        self.num_per_pose = num_per_pose
        self._running = True
        self.current_pose_index = 0
        self.captured_count = 0
        self.total_to_capture = len(poses_list) * num_per_pose
        
        # --- CONFIGURATION CLOUD ---
        self.task_id = f"CAP-{user_name}-{int(time.time())}"
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            self.table = self.dynamodb.Table('FaceAuthTasks')
        except Exception as e:
            print(f"Erreur initialisation AWS: {e}")

    def run(self):
        """Boucle principale de capture avec mises à jour Cloud."""
        # 1. Signaler le DEBUT de la capture sur AWS
        try:
            self.table.put_item(Item={
                'task_id': self.task_id,
                'status': 'capture_starting',
                'user': self.user_name,
                'progress': 0,
                'timestamp': str(time.time())
            })
        except: pass

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
                
                self.current_pose_index = pose_idx
                self.status_update.emit(f"Préparation pour la pose : {pose}")
                
                # --- UPDATE CLOUD : Changement de pose ---
                try:
                    self.table.update_item(
                        Key={'task_id': self.task_id},
                        UpdateExpression="set #s = :val",
                        ExpressionAttributeNames={'#s': 'status'},
                        ExpressionAttributeValues={':val': f'capturing_pose_{pose}'}
                    )
                except: pass

                # Attendre avant de commencer la capture
                for i in range(WAIT_BETWEEN_POSES):
                    if not self._running: break
                    self.status_update.emit(f"Pose '{pose}' commence dans {WAIT_BETWEEN_POSES - i}s...")
                    time.sleep(1)
                
                current_pose_count = 0
                last_save_time = 0
                last_centers = []
                
                while current_pose_count < self.num_per_pose and self._running:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    display_frame = frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    
                    if len(faces) == 1:
                        face_rect = faces[0]
                        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Check stability
                        cx, cy = x + w / 2, y + h / 2
                        last_centers.append((cx, cy))
                        if len(last_centers) > STABLE_FRAMES_REQUIRED: last_centers.pop(0)
                        
                        stable = False
                        if len(last_centers) == STABLE_FRAMES_REQUIRED:
                            max_dist = 0
                            for i in range(len(last_centers)):
                                for j in range(i+1, len(last_centers)):
                                    dist = math.hypot(last_centers[i][0]-last_centers[j][0], last_centers[i][1]-last_centers[j][1])
                                    if dist > max_dist: max_dist = dist
                            if max_dist <= STABILITY_THRESH: stable = True
                        
                        # Auto-capture
                        now = time.time()
                        if stable and (now - last_save_time >= MIN_TIME_BETWEEN_SAVES):
                            img_path = os.path.join(user_output_dir, f"{self.user_name}_{pose}_{current_pose_count:03d}.jpg")
                            cv2.imwrite(img_path, frame)
                            current_pose_count += 1
                            self.captured_count += 1
                            last_save_time = now
                            self.image_captured.emit(pose, img_path)
                            
                            prog = int((self.captured_count / self.total_to_capture) * 100)
                            self.progress_update.emit(prog)
                        
                        self.status_update.emit(f"Pose {pose}: {current_pose_count}/{self.num_per_pose}")
                    else:
                        self.status_update.emit("Placez votre visage au centre")
                    
                    # Preview
                    rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    qt_img = QImage(rgb_display.data, rgb_display.shape[1], rgb_display.shape[0], rgb_display.shape[1]*3, QImage.Format.Format_RGB888)
                    self.frame_ready.emit(qt_img)
                    time.sleep(0.03)
                
                if pose_idx < len(self.poses_list) - 1:
                    time.sleep(3)
        
        finally:
            cap.release()
            if self._running:
                self.status_update.emit(f"Capture terminée")
                # --- UPDATE CLOUD : Fin de capture ---
                try:
                    self.table.update_item(
                        Key={'task_id': self.task_id},
                        UpdateExpression="set #s = :val, progress = :p",
                        ExpressionAttributeNames={'#s': 'status'},
                        ExpressionAttributeValues={':val': 'capture_completed_locally', ':p': 100}
                    )
                except: pass
            else:
                self.status_update.emit("Capture annulée")

    def stop(self):
        self._running = False
        self.wait()

class CaptureDialog(QDialog):
    def __init__(self, user_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Enregistrement : {user_name}")
        self.setFixedSize(650, 750)
        
        # Style général (Dark Theme & Neon Blue)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f1222;
                border: 2px solid #1e293b;
            }
            QLabel {
                color: #e6eef8;
                font-family: 'Segoe UI', Arial;
            }
            #titleLabel {
                font-size: 22px;
                font-weight: bold;
                color: #7bdcff;
                margin-bottom: 5px;
            }
            #instructionLabel {
                font-size: 16px;
                color: #bcd3ff;
                background-color: rgba(58, 122, 254, 0.1);
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(58, 122, 254, 0.3);
            }
            QProgressBar {
                background-color: #141826;
                border: 1px solid #1e293b;
                border-radius: 10px;
                text-align: center;
                height: 20px;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                                    stop:0 #3a7afe, stop:1 #7bdcff);
                border-radius: 10px;
            }
            .PoseChip {
                background-color: #161b33;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 8px;
                color: #64748b;
                font-weight: bold;
                font-size: 11px;
            }
            .PoseChip[active="true"] {
                border: 1px solid #3a7afe;
                color: #3a7afe;
                background-color: rgba(58, 122, 254, 0.1);
            }
            .PoseChip[done="true"] {
                border: 1px solid #10b981;
                color: #10b981;
                background-color: rgba(16, 185, 129, 0.1);
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # En-tête
        header_layout = QVBoxLayout()
        self.title_label = QLabel(f"Profil : {user_name}")
        self.title_label.setObjectName("titleLabel")
        header_layout.addWidget(self.title_label)
        
        self.instruction_label = QLabel("Initialisation de la caméra...")
        self.instruction_label.setObjectName("instructionLabel")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.instruction_label)
        layout.addLayout(header_layout)

        # Zone Vidéo avec effet de lueur
        self.video_container = QFrame()
        self.video_container.setFixedSize(480, 480)
        self.video_container.setStyleSheet("""
            QFrame {
                border: 3px solid #1e293b;
                border-radius: 240px; /* Cercle parfait */
                background-color: #000000;
            }
        """)
        
        # Ajout d'une ombre portée bleue pour l'effet "Néon"
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(25)
        glow.setColor(QColor(58, 122, 254, 150))
        glow.setOffset(0, 0)
        self.video_container.setGraphicsEffect(glow)

        self.video_label = QLabel(self.video_container)
        self.video_label.setFixedSize(480, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 240px;") # Important pour clipper l'image

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_container)
        layout.addLayout(video_layout)

        # Indicateurs de Poses (Chips)
        self.poses_layout = QHBoxLayout()
        self.pose_chips = {}
        for p in ["Face", "Gauche", "Droite", "Haut", "Bas"]:
            chip = QLabel(p.upper())
            chip.setProperty("active", "false")
            chip.setProperty("done", "false")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setFixedWidth(90)
            chip.setObjectName(f"chip_{p.lower()}")
            chip.setProperty("class", "PoseChip")
            self.pose_chips[p.lower()] = chip
            self.poses_layout.addWidget(chip)
        layout.addLayout(self.poses_layout)

        # Barre de progression et Statut
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Veuillez vous placer face à l'objectif")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #64748b; font-size: 13px;")
        layout.addWidget(self.status_label)

        # Boutons de contrôle
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Terminer")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #059669; }
        """)
        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #dc2626; }
        """)
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_pose_status(self, current_pose):
        """Met à jour l'apparence des indicateurs de pose."""
        pose_map = {
            "frontale": "face", "gauche": "gauche", "droite": "droite", 
            "haut": "haut", "bas": "bas"
        }
        current_pose = pose_map.get(current_pose, "")

        for p_name, chip in self.pose_chips.items():
            if p_name == current_pose:
                chip.setProperty("active", "true")
                chip.setProperty("done", "false")
                self.instruction_label.setText(f"Action requise : Regardez vers {p_name.upper()}")
            elif chip.property("active") == "true": # Marquer comme fini si c'était l'ancien actif
                chip.setProperty("active", "false")
                chip.setProperty("done", "true")
            
            # Forcer le rafraîchissement du style CSS
            chip.style().unpolish(chip)
            chip.style().polish(chip)

    def set_frame(self, image):
        """Affiche l'image de la caméra dans le cercle."""
        # On crop l'image en carré pour le cercle si nécessaire
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatioByExpanding, 
            Qt.TransformationMode.SmoothTransformation
        ))
class QRCodeDialog(QDialog):
    def __init__(self, qr_data_bytes, trust_code, task_id, parent=None):
        super().__init__(parent)
        self.task_id = task_id 
        self.correct_code = trust_code.upper() # Le code généré
        self.time_left = 60 # 60 secondes

        self.setWindowTitle("Validation Double Facteur (MFA)")
        self.setFixedSize(450, 650)
        self.setStyleSheet("background-color: #0f172a; color: white;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        # 1. Chronomètre
        self.timer_label = QLabel(f"Temps restant : {self.time_left}s")
        self.timer_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #f87171;")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timer_label)

        # 2. QR Code
        self.qr_label = QLabel()
        qimg = QImage.fromData(qr_data_bytes)
        pixmap = QPixmap.fromImage(qimg)
        self.qr_label.setPixmap(pixmap.scaled(250, 250, Qt.AspectRatioMode.KeepAspectRatio))
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.qr_label)

        msg = QLabel("Scannez le code avec votre mobile et\nsaisissez le code de confiance affiché :")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("color: #94a3b8; font-size: 14px;")
        layout.addWidget(msg)

        # 3. Champ de saisie
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("TR-XXX-XXX")
        self.input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #1e293b;
                border: 2px solid #334155;
                border-radius: 10px;
                padding: 12px;
                font-size: 24px;
                font-weight: bold;
                color: #38bdf8;
                margin-top: 10px;
            }
        """)
        self.input_field.textChanged.connect(self.verify_code)
        layout.addWidget(self.input_field)

        # 4. Bouton de validation
        self.btn_next = QPushButton("Continuer vers la migration")
        self.btn_next.setEnabled(False) # Bloqué au début
        self.btn_next.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #64748b;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
            }
            QPushButton:enabled {
                background-color: #10b981;
                color: white;
            }
        """)
        self.btn_next.clicked.connect(self.accept)
        layout.addWidget(self.btn_next)

        self.setLayout(layout)

        # Lancer le compte à rebours
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)

    def update_timer(self):
        self.time_left -= 1
        self.timer_label.setText(f"Temps restant : {self.time_left}s")
        if self.time_left <= 0:
            self.timer.stop()
            self.reject() # Ferme la fenêtre si temps écoulé

    def verify_code(self):
        """Vérifie si le code saisi correspond au code généré."""
        typed = self.input_field.text().strip().upper()
        if typed == self.correct_code:
            self.btn_next.setEnabled(True)
            self.input_field.setStyleSheet("background-color: #064e3b; border: 2px solid #10b981; border-radius: 10px; padding: 12px; font-size: 24px; color: white;")
        else:
            self.btn_next.setEnabled(False)
            
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

    def show_qr_code(self, qr_bytes, trust_code):
        """Affiche le QR Code, cache l'appli principale et attend la validation mobile."""
        # On récupère le task_id actuel depuis le worker
        self.hide()
        current_task_id = self.auth_worker.task_id 
        authenticated_user = getattr(self.auth_worker, 'authenticated_user', 'Utilisateur')
        # On crée le dialogue en lui passant les 3 infos
        dialog = QRCodeDialog(qr_bytes, trust_code, current_task_id, self)
        
        # dialog.exec() bloque l'appli jusqu'à ce que self.accept() soit appelé (le scan réussi)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.status_label.setText("✅ Accès accordé ! Redirection...")
            # Ici tu peux ajouter une action finale (ouvrir un dossier, etc.)
            base_url = "https://ppppzr4jwi.execute-api.us-east-1.amazonaws.com/default/CloudVaultInterface"
            vault_url = f"{base_url}?user={authenticated_user}"            
            webbrowser.open(vault_url)
            QApplication.quit() 
        else:
            # Si l'utilisateur a fermé la fenêtre QR sans scanner, on réaffiche l'appli
            self.show()
      
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
            self.auth_worker.qr_signal.connect(self.show_qr_code)
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
    """Open a PyQt6 dialog for image capture with video preview."""
    dialog = CaptureDialog(user_name)
    
    poses = ["frontale", "gauche", "droite", "haut", "bas"]
    
    # Create and start the capture thread
    capture_thread = CaptureThread(user_name, poses, num_images_per_pose, parent=dialog)
    
    # Connect signals from thread to dialog
    capture_thread.frame_ready.connect(dialog.set_frame)
    capture_thread.status_update.connect(lambda txt: dialog.status_label.setText(txt))
    capture_thread.progress_update.connect(lambda val: dialog.progress_bar.setValue(val))
    capture_thread.image_captured.connect(lambda pose, path: dialog.update_pose_status(pose))
    
    # Start capture
    capture_thread.start()
    
    # Show dialog and wait for completion
    result = dialog.exec()
    
    # Stop thread if still running
    if capture_thread.isRunning():
        capture_thread.stop()
    
    if result == QDialog.DialogCode.Accepted:
        print(f"Capture d'images pour {user_name} terminée avec succès.")
    else:
        print("Capture annulée par l'utilisateur.")

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
    qr_signal = pyqtSignal(bytes, str)

    def __init__(self, known_encodings, known_names, known_poses, parent=None):
        super().__init__(parent)
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.known_poses = known_poses
        self._running = True
        
        # Configuration Cloud (à adapter avec tes noms)
        
        self.authenticated_user = None
        self.bucket_name = MON_BUCKET # <--- TON NOM DE BUCKET S3
        self.table_name = "FaceAuthTasks"
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table(self.table_name)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        self.task_id = str(uuid.uuid4()) 

        try:
            while self._running and not authentication_succeeded:
                ret, frame = cap.read()
                if not ret:
                    self.status_update.emit("Erreur lecture caméra")
                    break

                small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_small, 1)
                display = frame.copy()

                if self.authenticated_user is None:
                    if len(faces) > 0:
                        try:
                            enc = get_face_encoding(rgb_small, faces[0])
                            distances, indices = face_recognizer_knn.kneighbors([enc])
                            min_distance = distances[0][0]
                            idx = indices[0][0]
                            if min_distance < FACE_DETECTION_THRESHOLD:
                                self.authenticated_user = self.known_names[idx]
                                authenticated_user = self.authenticated_user
                                self.status_update.emit(f"Identifié: {authenticated_user}")
                                
                                self.table.put_item(Item={
                                    'task_id': self.task_id,
                                    'status': 'user_identified',
                                    'user': self.authenticated_user,
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
                    user_poses = self.known_poses.get(authenticated_user, {})
                    poses_list = list(user_poses.keys())
                    
                    if not poses_list:
                        self.status_update.emit("Aucune pose enregistrée pour ce utilisateur")
                        authenticated_user = None
                    else:
                        challenge_index = 0
                        while challenge_index < TOTAL_CHALLENGES and self._running:
                            required_pose = np.random.choice(poses_list)
                            self.status_update.emit(f"Défi {challenge_index+1}/{TOTAL_CHALLENGES} : {required_pose}")
                            
                            challenge_start = time.time()
                            hold_start = None
                            succeeded = False

                            while (time.time() - challenge_start) <= CHALLENGE_TIMEOUT and self._running:
                                ret2, frame2 = cap.read()
                                if not ret2: break
                                
                                small2 = cv2.resize(frame2, (0, 0), fx=0.75, fy=0.75)
                                gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
                                faces2 = detector(gray2, 1)

                                if len(faces2) > 0:
                                    face2 = faces2[0]
                                    face_rect_orig = dlib.rectangle(int(face2.left()/0.75), int(face2.top()/0.75), int(face2.right()/0.75), int(face2.bottom()/0.75))
                                    landmarks = predictor(frame2, face_rect_orig)
                                    yaw, pitch, roll, rvec, tvec = estimate_pose(landmarks, frame2.shape[1], frame2.shape[0])

                                    target = user_poses.get(required_pose)
                                    if target is None: break
                                    
                                    if abs(yaw - target[0]) < 20 and abs(pitch - target[1]) < 20:
                                        if hold_start is None: hold_start = time.time()
                                        elapsed = time.time() - hold_start
                                        self.progress_update.emit(int(min(100, (elapsed / POSE_DURATION) * 100)))
                                        if elapsed >= POSE_DURATION:
                                            succeeded = True
                                            break
                                    else:
                                        hold_start = None
                                        self.progress_update.emit(0)

                                rgb_disp = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                                qimg = QImage(rgb_disp.data, frame2.shape[1], frame2.shape[0], frame2.shape[1]*3, QImage.Format.Format_RGB888)
                                self.frame_ready.emit(qimg)
                                time.sleep(0.02)

                            if not succeeded:
                                self.status_update.emit("Défi échoué")
                                self.progress_update.emit(0)
                                self.authenticated_user = None
                                authenticated_user = None
                                try:
                                    self.table.update_item(
                                        Key={'task_id': self.task_id},
                                        UpdateExpression="set #s = :val, progress = :p",
                                        ExpressionAttributeNames={'#s': 'status'},
                                        ExpressionAttributeValues={':val': 'challenge_failed', ':p': 0}
                                    )
                                except: pass
                                
                                # Petite pause pour laisser l'utilisateur voir le message d'erreur
                                time.sleep(2.0) 
                                break 
                            else:
                                challenge_index += 1
                                self.table.update_item(
                                    Key={'task_id': self.task_id},
                                    UpdateExpression="set progress = :p",
                                    ExpressionAttributeValues={':p': 50}
                                )

                        if challenge_index >= TOTAL_CHALLENGES:
                            print("DEBUG: Défis réussis ! Tentative de génération QR...") # <--- AJOUTE ÇA
                            self.status_update.emit(f"Succès ! Sécurisation et MFA...")
                            
                            try:
                                # 1. Migration S3 (Déjà fait)
                                secret_data = f"Auth réussie pour {authenticated_user} le {time.ctime()}".encode()
                                payload, signature = encrypt_aes_512(secret_data)
                                s3_path = f"logs_authentification/{authenticated_user}_{int(time.time())}.bin"
                                upload_to_s3(payload, signature, self.bucket_name, s3_path)
                                
                                # 2. GÉNÉRATION DU QR CODE (NOUVEAU)
                                # On génère le trust_code et l'image du QR
                                print(f"DEBUG: Appel de generate_and_upload_qr pour {authenticated_user}") # <--- AJOUTE ÇA
                                trust_code, qr_bytes = generate_and_upload_qr(authenticated_user, self.bucket_name, self.task_id)
                                print(f"DEBUG: QR Généré avec succès. Code: {trust_code}")

                                
                                # 3. MISE À JOUR DYNAMODB AVEC LE TRUST_CODE
                                self.table.update_item(
                                    Key={'task_id': self.task_id},
                                    UpdateExpression="set #s = :val, progress = :p, trust_code = :tc",
                                    ExpressionAttributeNames={'#s': 'status'},
                                    ExpressionAttributeValues={
                                        ':val': 'mfa_pending', 
                                        ':p': 100,
                                        ':tc': trust_code
                                    }
                                )

                                # 4. ENVOI DU SIGNAL POUR AFFICHER LE QR DANS L'INTERFACE
                                print("DEBUG: Envoi du signal qr_signal...")
                                self.qr_signal.emit(qr_bytes, trust_code)
                                self.status_update.emit("✓ Authentifié. Scannez le QR Code.")

                            except Exception as e:
                                self.status_update.emit(f"Erreur Cloud: {e}")

                            time.sleep(2.0)
                            authentication_succeeded = True
                            break

                time.sleep(0.02)

        finally:
            cap.release()
            self.progress_update.emit(0)
            self.status_update.emit("Session terminée")

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