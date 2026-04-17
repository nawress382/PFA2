# app.py — Point d'entrée principal de l'application PFA2
# Double-clic sur PFA2_SecureCloud.exe pour lancer sans terminal

import sys
import os


def _load_env():
    """
    Charge les variables d'environnement depuis le fichier .env
    situé à côté de l'exécutable ou du script.
    """
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    env_path = os.path.join(base_dir, ".env")

    if not os.path.exists(env_path):
        env_path = os.path.join(os.getcwd(), ".env")

    if os.path.exists(env_path):
        print(f"[Config] Chargement .env : {env_path}")
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        print("[Config] Variables Azure chargées ✓")
    else:
        print("[Config] .env introuvable — variables système utilisées")


# Charger les variables AVANT tout import Azure
_load_env()

from PyQt6.QtWidgets import QApplication
from face_pose_estimation import FaceAuthApp
from migrate_gui import MigrateApp


class AppController:
    """
    Contrôleur principal :
    1. Lance FaceAuthApp  (Authentification biométrique)
    2. Après succès → ouvre MigrateApp (Migration Azure)
    """

    def __init__(self, qt_app: QApplication):
        self.qt_app         = qt_app
        self.auth_window    = None
        self.migrate_window = None

    def start(self):
        self.auth_window = FaceAuthApp()

        # Injecter le callback d'ouverture de migration
        def patched_on_auth_finished():
            status = self.auth_window.status_label.text().lower()
            auth_success = any(k in status for k in [
                "authentifié", "azure", "réussie", "succès", "sécurisé"
            ])

            # Nettoyer le worker
            try:
                self.auth_window.auth_worker.quit()
                self.auth_window.auth_worker.wait(1000)
            except Exception:
                pass
            self.auth_window.auth_worker = None

            if auth_success:
                self._open_migrate()
            else:
                self.auth_window.status_label.setText("Authentification terminée")
                self.auth_window.progress_bar.setValue(0)

        self.auth_window._on_auth_finished = patched_on_auth_finished
        self.auth_window.show()

    def _open_migrate(self):
        """Ferme l'authentification et ouvre la migration."""
        if self.auth_window:
            try:
                self.auth_window.timer.stop()
                self.auth_window.cap.release()
            except Exception:
                pass
            self.auth_window.close()
            self.auth_window = None

        self.migrate_window = MigrateApp()
        self.migrate_window.show()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PFA2 — Authentification & Migration Sécurisée")

    controller = AppController(app)
    controller.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()