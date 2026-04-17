# ══════════════════════════════════════════════════════════════════════════════
# trust_qr_module.py — code caché + saisie obligatoire du code scanné
# ══════════════════════════════════════════════════════════════════════════════

import time
import uuid
import hashlib
import qrcode
import io

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer


# ══════════════════════════════════════════════════════════════════════════════
# 1. GÉNÉRATION DU CODE DE CONFIANCE
# ══════════════════════════════════════════════════════════════════════════════

def generate_trust_code(user_name: str) -> dict:
    task_id    = str(uuid.uuid4())
    timestamp  = time.time()
    raw        = f"{user_name}|{task_id}|{timestamp}"
    signature  = hashlib.sha256(raw.encode()).hexdigest()[:6].upper()
    trust_code = f"TR-{signature[:3]}-{signature[3:]}"
    return {
        "trust_code": trust_code,
        "task_id":    task_id,
        "qr_payload": trust_code,
        "timestamp":  timestamp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. GÉNÉRATION DE L'IMAGE QR-CODE
# ══════════════════════════════════════════════════════════════════════════════

def generate_qr_image(payload: str, size: int = 340) -> QPixmap:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=20,
        border=6,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#000000", back_color="#FFFFFF")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    qimg = QImage()
    qimg.loadFromData(buf.read())
    return QPixmap.fromImage(qimg).scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. STYLES
# ══════════════════════════════════════════════════════════════════════════════

DIALOG_STYLE = """
QDialog {
    background-color: #F7F8FC;
    font-family: 'Segoe UI', 'Calibri', sans-serif;
}
QFrame#headerFrame {
    background-color: #1A2340;
}
QLabel#dlgMainTitle {
    color: #F1F5F9;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#dlgSubtitle {
    color: #94A3B8;
    font-size: 11px;
    letter-spacing: 2px;
}
QFrame#qrCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}
QLabel#trustLabel {
    color: #64748B;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
}
QLabel#infoLabel {
    color: #64748B;
    font-size: 12px;
}
QLabel#userLabel {
    color: #2563EB;
    font-size: 14px;
    font-weight: 700;
}
QLabel#timerLabel {
    color: #DC2626;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#btnContinue {
    background-color: #94A3B8;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    padding: 12px 32px;
}
QPushButton#btnContinue:enabled {
    background-color: #2563EB;
}
QPushButton#btnContinue:enabled:hover {
    background-color: #1D4ED8;
}
"""

STYLE_INPUT_DEFAULT = """
QLineEdit {
    background-color: #F8FAFF;
    border: 2px solid #BFDBFE;
    border-radius: 8px;
    padding: 10px;
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 6px;
    color: #1A2340;
    font-family: 'Courier New', monospace;
}
QLineEdit:focus { border: 2px solid #2563EB; }
"""

STYLE_INPUT_OK = """
QLineEdit {
    background-color: #ECFDF5;
    border: 2px solid #059669;
    border-radius: 8px;
    padding: 10px;
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 6px;
    color: #059669;
    font-family: 'Courier New', monospace;
}
"""

STYLE_INPUT_ERR = """
QLineEdit {
    background-color: #FFF1F2;
    border: 2px solid #DC2626;
    border-radius: 8px;
    padding: 10px;
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 6px;
    color: #DC2626;
    font-family: 'Courier New', monospace;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIALOGUE
# ══════════════════════════════════════════════════════════════════════════════

class TrustQRDialog(QDialog):
    """
    - Code de confiance CACHÉ (l'utilisateur doit scanner pour l'obtenir)
    - Saisie du code OBLIGATOIRE → bouton activé seulement si code correct
    - Expiration → reject() sans ouvrir la migration
    """

    def __init__(self, user_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Code de Confiance — Session Sécurisée")
        self.setFixedSize(540, 740)
        self.setStyleSheet(DIALOG_STYLE)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        self._expired  = False
        self._accepted = False
        self.session_data = generate_trust_code(user_name)
        self.remaining    = 60

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("headerFrame")
        header.setFixedHeight(90)
        hl = QVBoxLayout()
        hl.setContentsMargins(28, 16, 28, 16)
        hl.setSpacing(3)
        title = QLabel("🔐  SESSION SÉCURISÉE")
        title.setObjectName("dlgMainTitle")
        sub = QLabel("CODE DE CONFIANCE GÉNÉRÉ PAR AZURE CLOUD")
        sub.setObjectName("dlgSubtitle")
        hl.addWidget(title)
        hl.addWidget(sub)
        header.setLayout(hl)
        layout.addWidget(header)

        # ── Corps ─────────────────────────────────────────────────────────
        body = QVBoxLayout()
        body.setContentsMargins(24, 14, 24, 14)
        body.setSpacing(10)

        # Utilisateur + Timer
        user_row = QHBoxLayout()
        user_icon = QLabel("👤")
        user_icon.setStyleSheet("font-size: 20px;")
        user_info = QVBoxLayout()
        user_info.setSpacing(1)
        lbl = QLabel("UTILISATEUR AUTHENTIFIÉ")
        lbl.setObjectName("trustLabel")
        name_lbl = QLabel(user_name)
        name_lbl.setObjectName("userLabel")
        user_info.addWidget(lbl)
        user_info.addWidget(name_lbl)
        user_row.addWidget(user_icon)
        user_row.addSpacing(8)
        user_row.addLayout(user_info)
        user_row.addStretch()
        self.timer_label = QLabel("⏱  Expire dans : 60s")
        self.timer_label.setObjectName("timerLabel")
        user_row.addWidget(self.timer_label)
        body.addLayout(user_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #E2E8F0; max-height: 1px;")
        body.addWidget(sep)

        # ── Carte QR ──────────────────────────────────────────────────────
        qr_card = QFrame()
        qr_card.setObjectName("qrCard")
        qr_layout = QVBoxLayout()
        qr_layout.setContentsMargins(16, 14, 16, 14)
        qr_layout.setSpacing(10)
        qr_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # QR-Code — le code est DANS le QR, pas affiché en clair
        qr_pixmap = generate_qr_image(self.session_data["qr_payload"], size=340)
        qr_img_label = QLabel()
        qr_img_label.setPixmap(qr_pixmap)
        qr_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qr_img_label.setStyleSheet(
            "background-color: #FFFFFF;"
            "padding: 10px;"
            "border: 2px solid #E2E8F0;"
            "border-radius: 8px;"
        )
        qr_layout.addWidget(qr_img_label)

        scan_info = QLabel("📱  Scannez ce QR-Code avec votre téléphone")
        scan_info.setObjectName("infoLabel")
        scan_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qr_layout.addWidget(scan_info)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #F1F5F9; max-height: 1px;")
        qr_layout.addWidget(sep2)

        # ── Saisie obligatoire — PAS d'affichage du code en clair ─────────
        verify_lbl = QLabel("ENTREZ LE CODE OBTENU PAR LE SCAN")
        verify_lbl.setObjectName("trustLabel")
        verify_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        qr_layout.addWidget(verify_lbl)

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("Ex : TR-073-E48")
        self.code_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.code_input.setStyleSheet(STYLE_INPUT_DEFAULT)
        self.code_input.setMaxLength(11)
        self.code_input.textChanged.connect(self._check_code)
        qr_layout.addWidget(self.code_input)

        self.verify_status = QLabel("")
        self.verify_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verify_status.setStyleSheet("font-size: 12px; font-weight: 600;")
        qr_layout.addWidget(self.verify_status)

        qr_card.setLayout(qr_layout)
        body.addWidget(qr_card)

        # ── Bouton continuer (désactivé par défaut) ───────────────────────
        self.btn_continue = QPushButton("✓   Continuer vers la migration")
        self.btn_continue.setObjectName("btnContinue")
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self._on_continue_clicked)
        body.addWidget(self.btn_continue, alignment=Qt.AlignmentFlag.AlignCenter)

        body_widget = QFrame()
        body_widget.setLayout(body)
        layout.addWidget(body_widget)
        self.setLayout(layout)

        # Timer 60s
        self._countdown = QTimer()
        self._countdown.timeout.connect(self._tick)
        self._countdown.start(1000)

    def _check_code(self, text: str):
        entered  = text.strip().upper()
        expected = self.session_data["trust_code"]

        if entered == expected:
            self.code_input.setStyleSheet(STYLE_INPUT_OK)
            self.btn_continue.setEnabled(True)
            self.verify_status.setText("✅  Code correct — migration autorisée")
            self.verify_status.setStyleSheet(
                "font-size: 12px; font-weight: 600; color: #059669;")
        elif len(entered) == 0:
            self.code_input.setStyleSheet(STYLE_INPUT_DEFAULT)
            self.btn_continue.setEnabled(False)
            self.verify_status.setText("")
        else:
            self.code_input.setStyleSheet(STYLE_INPUT_ERR)
            self.btn_continue.setEnabled(False)
            self.verify_status.setText("❌  Code incorrect — vérifiez le scan")
            self.verify_status.setStyleSheet(
                "font-size: 12px; font-weight: 600; color: #DC2626;")

    def _on_continue_clicked(self):
        self._accepted = True
        self._countdown.stop()
        self.accept()

    def _tick(self):
        self.remaining -= 1
        self.timer_label.setText(f"⏱  Expire dans : {self.remaining}s")
        if self.remaining <= 0:
            self._countdown.stop()
            self._expired = True
            self.reject()

    def is_expired(self) -> bool:
        return self._expired

    def get_session_data(self) -> dict:
        return self.session_data