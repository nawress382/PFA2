# migrate_gui.py — Interface graphique de migration sécurisée vers Azure
import os
import sys
import time
import threading
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QLineEdit, QProgressBar, QTextEdit, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor

from migrate_document import migrate_document


# ── Signaux pour communication thread → UI ─────────────────────────────────
class MigrationSignals(QObject):
    log     = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)


# ══════════════════════════════════════════════════════════════════════════════
# FENÊTRE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class MigrateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Migration Sécurisée vers Azure Cloud — PFA2")
        self.setFixedSize(720, 600)
        self.selected_file = None

        self.setStyleSheet("""
            QWidget {
                background-color: #0f1222;
                color: #e6eef8;
                font-family: 'Segoe UI', Arial;
            }
            QLabel#title {
                font-size: 20px;
                font-weight: bold;
                color: #7bdcff;
            }
            QLabel#subtitle {
                font-size: 12px;
                color: #64748b;
            }
            QLabel#sectionLabel {
                font-size: 13px;
                color: #bcd3ff;
                font-weight: bold;
            }
            QLineEdit {
                background-color: #141826;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 10px;
                color: #e6eef8;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #3a7afe;
            }
            QPushButton#browseBtn {
                background-color: #1e293b;
                color: #7bdcff;
                border: 1px solid #3a7afe;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton#browseBtn:hover {
                background-color: #3a7afe;
                color: white;
            }
            QPushButton#migrateBtn {
                background-color: #3a7afe;
                color: white;
                border-radius: 10px;
                padding: 14px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton#migrateBtn:hover { background-color: #5790ff; }
            QPushButton#migrateBtn:disabled {
                background-color: #1e293b;
                color: #64748b;
            }
            QProgressBar {
                background-color: #141826;
                border: 1px solid #1e293b;
                border-radius: 8px;
                height: 18px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3a7afe, stop:1 #7bdcff
                );
                border-radius: 8px;
            }
            QTextEdit {
                background-color: #0a0e1a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                color: #7bdcff;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 8px;
            }
            QFrame#separator {
                background-color: #1e293b;
                max-height: 1px;
            }
        """)

        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout()
        main.setContentsMargins(30, 30, 30, 30)
        main.setSpacing(18)

        # ── Titre ──────────────────────────────────────────────────────────
        title = QLabel("Migration Sécurisée vers Azure Cloud")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Chiffrement AES-512   •  Signature HMAC-512  •  Azure Blob Storage")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main.addWidget(title)
        main.addWidget(subtitle)

        # ── Séparateur ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        main.addWidget(sep)

        # ── Sélection du fichier ───────────────────────────────────────────
        file_label = QLabel("Fichier à migrer")
        file_label.setObjectName("sectionLabel")
        main.addWidget(file_label)

        file_row = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Cliquez sur Parcourir pour sélectionner un fichier...")
        self.file_input.setReadOnly(True)

        browse_btn = QPushButton("📁  Parcourir")
        browse_btn.setObjectName("browseBtn")
        browse_btn.setFixedWidth(130)
        browse_btn.clicked.connect(self.browse_file)

        file_row.addWidget(self.file_input)
        file_row.addWidget(browse_btn)
        main.addLayout(file_row)

        # ── Nom du propriétaire ────────────────────────────────────────────
        owner_label = QLabel("Votre nom")
        owner_label.setObjectName("sectionLabel")
        main.addWidget(owner_label)

        self.owner_input = QLineEdit()
        self.owner_input.setPlaceholderText("Ex: nawres")
        self.owner_input.setText("nawres")
        main.addWidget(self.owner_input)

        # ── Informations sur le fichier ────────────────────────────────────
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #64748b; font-size: 12px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.info_label)

        # ── Bouton Migration ───────────────────────────────────────────────
        self.migrate_btn = QPushButton("🚀  Lancer la Migration Sécurisée")
        self.migrate_btn.setObjectName("migrateBtn")
        self.migrate_btn.setEnabled(False)
        self.migrate_btn.clicked.connect(self.start_migration)
        main.addWidget(self.migrate_btn)

        # ── Barre de progression ───────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setValue(0)
        main.addWidget(self.progress)

        # ── Journal (logs) ─────────────────────────────────────────────────
        log_label = QLabel("Journal de migration")
        log_label.setObjectName("sectionLabel")
        main.addWidget(log_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(160)
        self.log_box.setPlaceholderText("Les étapes de migration apparaîtront ici...")
        main.addWidget(self.log_box)

        # ── Statut final ───────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main.addWidget(self.status_label)

        self.setLayout(main)

    # ── Actions ────────────────────────────────────────────────────────────

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner un fichier à migrer",
            os.path.expanduser("~\\Documents"),
            "Tous les fichiers (*.*)"
        )
        if file_path:
            self.selected_file = file_path
            self.file_input.setText(file_path)

            # Infos sur le fichier
            size_kb = os.path.getsize(file_path) / 1024
            name    = os.path.basename(file_path)
            self.info_label.setText(f"📄 {name}  —  {size_kb:.1f} KB")

            self.migrate_btn.setEnabled(True)
            self.status_label.setText("")
            self.log_box.clear()

    def start_migration(self):
        if not self.selected_file:
            return

        owner = self.owner_input.text().strip() or "nawres"

        # Désactiver le bouton pendant la migration
        self.migrate_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_box.clear()
        self.status_label.setText("")

        # Créer les signaux
        self.signals = MigrationSignals()
        self.signals.log.connect(self.append_log)
        self.signals.progress.connect(self.progress.setValue)
        self.signals.finished.connect(self.on_finished)

        # Lancer dans un thread pour ne pas bloquer l'UI
        thread = threading.Thread(
            target=self._run_migration,
            args=(self.selected_file, owner),
            daemon=True
        )
        thread.start()

    def _run_migration(self, file_path, owner):
        """Exécuté dans un thread séparé."""
        try:
            self.signals.log.emit("🔍 Lecture du fichier...")
            self.signals.progress.emit(10)
            time.sleep(0.3)

            self.signals.log.emit("🔐 Chiffrement AES-256 double couche...")
            self.signals.progress.emit(30)
            time.sleep(0.3)

            self.signals.log.emit("✍️  Signature HMAC-512...")
            self.signals.progress.emit(50)
            time.sleep(0.3)

            self.signals.log.emit("☁️  Upload vers Azure Blob Storage...")
            self.signals.progress.emit(70)

            result = migrate_document(file_path, owner)

            self.signals.progress.emit(90)
            self.signals.log.emit("📋 Enregistrement du log d'audit...")
            time.sleep(0.3)

            self.signals.progress.emit(100)
            self.signals.log.emit(f"✅ Fichier chiffré : {result['blob_path']}")
            self.signals.log.emit(f"📋 Log d'audit     : {result['log_path']}")
            self.signals.log.emit(f"🔑 Hash SHA-256    : {result['original_hash'][:32]}...")
            self.signals.finished.emit(True, "Migration réussie !")

        except Exception as e:
            self.signals.log.emit(f"❌ Erreur : {e}")
            self.signals.finished.emit(False, str(e))

    def append_log(self, message: str):
        self.log_box.append(f"  {message}")

    def on_finished(self, success: bool, message: str):
        self.migrate_btn.setEnabled(True)
        if success:
            self.status_label.setText("✅ Migration réussie — Fichier sécurisé sur Azure !")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #10b981;")
        else:
            self.status_label.setText(f"❌ Erreur : {message}")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ef4444;")


# ── Point d'entrée ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = MigrateApp()
    window.show()
    sys.exit(app.exec())