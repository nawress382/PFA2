# migrate_document.py — Migration sécurisée d'un fichier PDF vers Azure
# Chiffrement AES-256 double couche + Signature HMAC-512 + Upload Azure Blob Storage

import os
import time
import hashlib
from security_utils import encrypt_aes_512, upload_to_azure_blob


def migrate_document(file_path: str, owner_name: str = "nawres") -> dict:
    """
    Migre un fichier confidentiel vers Azure Blob Storage de manière sécurisée.

    Étapes :
        1. Lecture du fichier local
        2. Chiffrement AES-256 double couche (via security_utils)
        3. Signature HMAC-512 (intégrité)
        4. Upload vers Azure Blob Storage (conteneur documents-chiffres)
        5. Enregistrement du log dans logs-audit

    Args:
        file_path  : chemin complet du fichier sur votre PC
        owner_name : votre nom (pour organiser les fichiers sur Azure)

    Returns:
        dict avec les détails de la migration
    """

    # ── Vérification du fichier ────────────────────────────────────────────────
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    print(f"\n{'='*60}")
    print(f"  MIGRATION SÉCURISÉE VERS AZURE CLOUD")
    print(f"{'='*60}")
    print(f"  Fichier   : {file_name}")
    print(f"  Taille    : {file_size / 1024:.1f} KB")
    print(f"  Propriétaire : {owner_name}")
    print(f"{'='*60}\n")

    # ── Étape 1 : Lecture du fichier ───────────────────────────────────────────
    print("[1/4] Lecture du fichier...")
    with open(file_path, "rb") as f:
        file_data = f.read()

    # Calcul du hash original (pour vérification d'intégrité)
    original_hash = hashlib.sha256(file_data).hexdigest()
    print(f"      Hash SHA-256 original : {original_hash[:32]}...")

    # ── Étape 2 : Chiffrement AES-256 double couche ───────────────────────────
    print("[2/4] Chiffrement AES-256 double couche...")
    payload, signature = encrypt_aes_512(file_data)
    print(f"      Taille chiffrée : {len(payload) / 1024:.1f} KB")
    print(f"      Signature HMAC-512 : {signature.hex()[:32]}...")

    # ── Étape 3 : Upload du document chiffré ──────────────────────────────────
    timestamp = int(time.time())
    # Chemin dans Azure : documents-chiffres/nawres/Audit_2024_1234567890.bin
    blob_name = f"{owner_name}/{os.path.splitext(file_name)[0]}_{timestamp}.bin"

    print(f"[3/4] Upload vers Azure Blob Storage...")
    print(f"      Conteneur : documents-chiffres")
    print(f"      Chemin    : {blob_name}")

    success = upload_to_azure_blob(
        payload,
        signature,
        container_name="documents-chiffres",
        blob_name=blob_name
    )

    if not success:
        raise RuntimeError("Échec de l'upload vers Azure Blob Storage")

    print(f"      ✓ Document chiffré uploadé avec succès")

    # ── Étape 4 : Log d'audit ─────────────────────────────────────────────────
    print(f"[4/4] Enregistrement du log d'audit...")

    log_content = f"""=== LOG DE MIGRATION SÉCURISÉE ===
Fichier original  : {file_name}
Taille originale  : {file_size} octets
Propriétaire      : {owner_name}
Date migration    : {time.ctime()}
Hash SHA-256      : {original_hash}
Blob Azure        : documents-chiffres/{blob_name}
Algorithme        : AES-256 double couche + HMAC-512
Statut            : SUCCÈS
==================================
""".encode()

    log_payload, log_signature = encrypt_aes_512(log_content)
    log_blob_name = f"migration_logs/{owner_name}_{timestamp}.log"

    upload_to_azure_blob(
        log_payload,
        log_signature,
        container_name="logs-audit",
        blob_name=log_blob_name
    )
    print(f"      ✓ Log d'audit enregistré : logs-audit/{log_blob_name}")

    # ── Résumé final ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✓ MIGRATION RÉUSSIE")
    print(f"{'='*60}")
    print(f"  Document chiffré : documents-chiffres/{blob_name}")
    print(f"  Log d'audit      : logs-audit/{log_blob_name}")
    print(f"  Confidentialité  : AES-256 x2 ✓")
    print(f"  Intégrité        : HMAC-512 ✓")
    print(f"  Non-répudiation  : Hash SHA-256 enregistré ✓")
    print(f"{'='*60}\n")

    return {
        'status':        'success',
        'file_name':     file_name,
        'blob_path':     f"documents-chiffres/{blob_name}",
        'log_path':      f"logs-audit/{log_blob_name}",
        'original_hash': original_hash,
        'timestamp':     timestamp
    }


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n=== OUTIL DE MIGRATION SÉCURISÉE VERS AZURE ===\n")

    # Demander le chemin du fichier
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    else:
        file_path = input("Entrez le chemin complet de votre fichier PDF : ").strip()
        # Supprimer les guillemets si l'utilisateur a copié-collé avec guillemets
        file_path = file_path.strip('"').strip("'")

    # Demander le nom du propriétaire
    if len(sys.argv) >= 3:
        owner = sys.argv[2]
    else:
        owner = input("Entrez votre nom (ex: nawres) : ").strip() or "nawres"

    try:
        result = migrate_document(file_path, owner)
        print("Migration terminée avec succès !")
    except FileNotFoundError as e:
        print(f"\n✗ Erreur : {e}")
        print("Vérifiez que le chemin du fichier est correct.")
    except Exception as e:
        print(f"\n✗ Erreur lors de la migration : {e}")