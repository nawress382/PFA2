# dataset_sync.py — Synchronisation du dataset local ↔ Azure Blob Storage
import os
import cv2
import time
from azure.storage.blob import BlobServiceClient

# ── Configuration ─────────────────────────────────────────────────────────────
AZURE_CONTAINER_DATASET = "documents-chiffres"
DATASET_DIR             = "dataset"


def _get_blob_service() -> BlobServiceClient:
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING non définie.")
    return BlobServiceClient.from_connection_string(connection_string)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD : Local → Azure Blob Storage
# ══════════════════════════════════════════════════════════════════════════════

def upload_dataset_to_azure(dataset_dir=DATASET_DIR, progress_callback=None) -> dict:
    """
    Upload tout le dataset local vers Azure Blob Storage.
    Structure dans le conteneur :
        dataset/<user_name>/<image>.jpg

    Args:
        dataset_dir       : chemin local du dossier dataset
        progress_callback : fonction(status: str, info: dict) optionnelle

    Returns:
        dict avec le nombre de fichiers uploadés et les erreurs
    """
    if not os.path.exists(dataset_dir):
        print(f"[Dataset Sync] Dossier '{dataset_dir}' introuvable.")
        return {'uploaded': 0, 'errors': 0}

    try:
        blob_service = _get_blob_service()
    except Exception as e:
        print(f"[Dataset Sync] Erreur connexion Azure: {e}")
        return {'uploaded': 0, 'errors': 1}

    uploaded = 0
    errors   = 0

    # Parcourir tous les utilisateurs et leurs images
    for user_name in os.listdir(dataset_dir):
        user_path = os.path.join(dataset_dir, user_name)
        if not os.path.isdir(user_path):
            continue

        print(f"\n[Dataset Sync] Upload du dataset de : {user_name}")

        for img_name in os.listdir(user_path):
            if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
                continue

            local_path = os.path.join(user_path, img_name)
            # Chemin dans Azure Blob : dataset/Mariem/Mariem_frontale_001.jpg
            blob_name  = f"dataset/{user_name}/{img_name}"

            try:
                blob_client = blob_service.get_blob_client(
                    container=AZURE_CONTAINER_DATASET,
                    blob=blob_name
                )
                with open(local_path, "rb") as f:
                    blob_client.upload_blob(f, overwrite=True)

                uploaded += 1
                print(f"  ✓ {blob_name}")

                if progress_callback:
                    progress_callback('image_uploaded', {
                        'user': user_name,
                        'file': img_name,
                        'blob': blob_name
                    })

            except Exception as e:
                errors += 1
                print(f"  ✗ Erreur pour {img_name}: {e}")

    print(f"\n[Dataset Sync] Upload terminé : {uploaded} images uploadées, {errors} erreurs.")
    return {'uploaded': uploaded, 'errors': errors}


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD : Azure Blob Storage → Local
# ══════════════════════════════════════════════════════════════════════════════

def download_dataset_from_azure(dataset_dir=DATASET_DIR, progress_callback=None) -> dict:
    """
    Télécharge le dataset depuis Azure Blob Storage vers le dossier local.
    Utile si le dossier local est vide ou sur une nouvelle machine.

    Args:
        dataset_dir       : chemin local de destination
        progress_callback : fonction(status: str, info: dict) optionnelle

    Returns:
        dict avec le nombre de fichiers téléchargés et les erreurs
    """
    try:
        blob_service = _get_blob_service()
    except Exception as e:
        print(f"[Dataset Sync] Erreur connexion Azure: {e}")
        return {'downloaded': 0, 'errors': 1}

    container_client = blob_service.get_container_client(AZURE_CONTAINER_DATASET)

    downloaded = 0
    errors     = 0

    print(f"\n[Dataset Sync] Téléchargement du dataset depuis Azure...")

    # Lister tous les blobs qui commencent par "dataset/"
    blobs = list(container_client.list_blobs(name_starts_with="dataset/"))

    if not blobs:
        print("[Dataset Sync] Aucune image trouvée dans Azure.")
        return {'downloaded': 0, 'errors': 0}

    for blob in blobs:
        # blob.name = "dataset/Mariem/Mariem_frontale_001.jpg"
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue

        user_name = parts[1]
        img_name  = parts[2]

        local_user_dir = os.path.join(dataset_dir, user_name)
        os.makedirs(local_user_dir, exist_ok=True)
        local_path = os.path.join(local_user_dir, img_name)

        # Ne pas re-télécharger si le fichier existe déjà
        if os.path.exists(local_path):
            continue

        try:
            blob_client   = blob_service.get_blob_client(
                container=AZURE_CONTAINER_DATASET,
                blob=blob.name
            )
            data = blob_client.download_blob().readall()
            with open(local_path, "wb") as f:
                f.write(data)

            downloaded += 1
            print(f"  ✓ {blob.name}")

            if progress_callback:
                progress_callback('image_downloaded', {
                    'user': user_name,
                    'file': img_name,
                    'local_path': local_path
                })

        except Exception as e:
            errors += 1
            print(f"  ✗ Erreur pour {blob.name}: {e}")

    print(f"\n[Dataset Sync] Téléchargement terminé : {downloaded} images, {errors} erreurs.")
    return {'downloaded': downloaded, 'errors': errors}


# ══════════════════════════════════════════════════════════════════════════════
# SYNC AUTOMATIQUE AU DÉMARRAGE
# ══════════════════════════════════════════════════════════════════════════════

def sync_dataset_on_startup(dataset_dir=DATASET_DIR) -> str:
    """
    Appelé au démarrage de l'application.
    - Si le dataset local est VIDE  → télécharge depuis Azure
    - Si le dataset local existe    → upload vers Azure pour le sauvegarder

    Returns:
        Message de statut
    """
    local_is_empty = (
        not os.path.exists(dataset_dir) or
        not any(
            os.path.isdir(os.path.join(dataset_dir, d))
            for d in os.listdir(dataset_dir)
        ) if os.path.exists(dataset_dir) else True
    )

    if local_is_empty:
        print("[Dataset Sync] Dataset local vide → Téléchargement depuis Azure...")
        result = download_dataset_from_azure(dataset_dir)
        if result['downloaded'] > 0:
            return f"✓ {result['downloaded']} images téléchargées depuis Azure"
        else:
            return "⚠ Aucune image trouvée sur Azure — commencez par capturer"
    else:
        print("[Dataset Sync] Dataset local détecté → Upload vers Azure...")
        result = upload_dataset_to_azure(dataset_dir)
        return f"✓ Dataset synchronisé ({result['uploaded']} images sur Azure)"


# ══════════════════════════════════════════════════════════════════════════════
# TEST EN LIGNE DE COMMANDE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python dataset_sync.py upload    — Upload le dataset local vers Azure")
        print("  python dataset_sync.py download  — Télécharge le dataset depuis Azure")
        print("  python dataset_sync.py sync      — Sync automatique au démarrage")
        sys.exit(0)

    commande = sys.argv[1].lower()

    if commande == "upload":
        upload_dataset_to_azure()
    elif commande == "download":
        download_dataset_from_azure()
    elif commande == "sync":
        msg = sync_dataset_on_startup()
        print(f"\nRésultat: {msg}")
    else:
        print(f"Commande inconnue: {commande}")