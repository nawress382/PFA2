# security_utils.py — Adapté pour Microsoft Azure
import hashlib
import hmac
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# ── Azure SDK ──────────────────────────────────────────────────────────────────
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# ── Récupération des clés depuis Azure Key Vault ───────────────────────────────
def _get_master_key() -> bytes:
    """
    Récupère la clé AES depuis Azure Key Vault.
    Si Key Vault est inaccessible (mode local/dev), utilise une clé locale de fallback.
    """
    keyvault_uri = os.environ.get("KEYVAULT_URI", "https://pfa2-keyvault-auth.vault.azure.net/")
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=keyvault_uri, credential=credential)
        secret_value = client.get_secret("AES-SECRET-KEY").value
        # La clé stockée dans Key Vault est une chaîne hex de 128 caractères = 64 octets
        return bytes.fromhex(secret_value)
    except Exception as e:
        print(f"[Key Vault] Impossible de récupérer la clé AES: {e}")
        print("[Key Vault] Utilisation de la clé locale de fallback (développement uniquement)")
        # Fallback local pour le développement
        MON_TEXTE_SECRET = "Ceci est mon mot de passe secret pour le PFE"
        return hashlib.sha512(MON_TEXTE_SECRET.encode()).digest()


def _get_hmac_key() -> bytes:
    """
    Récupère la clé HMAC depuis Azure Key Vault.
    """
    keyvault_uri = os.environ.get("KEYVAULT_URI", "https://pfa2-keyvault-auth.vault.azure.net/")
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=keyvault_uri, credential=credential)
        secret_value = client.get_secret("HMAC-SECRET-KEY").value
        return bytes.fromhex(secret_value)
    except Exception as e:
        print(f"[Key Vault] Impossible de récupérer la clé HMAC: {e}")
        # Fallback : utiliser la même clé master pour HMAC
        return _get_master_key()


# ── Chiffrement AES double couche (AES-256 x2 = "AES-512 conceptuel") ─────────
def encrypt_aes_512(data: bytes):
    """
    Chiffre les données avec deux couches AES-256 (CBC).
    Les clés sont récupérées depuis Azure Key Vault.
    
    Retourne : (payload_chiffré, signature_hmac)
    """
    master_key = _get_master_key()
    hmac_key   = _get_hmac_key()

    key1 = master_key[:32]   # 32 octets = 256 bits
    key2 = master_key[32:]   # 32 octets = 256 bits

    # --- COUCHE 1 ---
    iv1 = os.urandom(16)
    cipher1 = AES.new(key1, AES.MODE_CBC, iv1)
    layer1_data = cipher1.encrypt(pad(data, AES.block_size))

    # --- COUCHE 2 ---
    iv2 = os.urandom(16)
    cipher2 = AES.new(key2, AES.MODE_CBC, iv2)
    layer2_data = cipher2.encrypt(pad(layer1_data, AES.block_size))

    # --- SIGNATURE HMAC-512 ---
    final_payload = iv1 + iv2 + layer2_data
    signature = hmac.new(hmac_key, final_payload, hashlib.sha512).digest()

    return final_payload, signature


def decrypt_aes_512(payload: bytes, signature: bytes) -> bytes:
    """
    Vérifie la signature HMAC puis déchiffre les données.
    Lève une exception si la signature est invalide (intégrité compromise).
    """
    hmac_key   = _get_hmac_key()
    master_key = _get_master_key()

    # Vérification HMAC
    expected_sig = hmac.new(hmac_key, payload, hashlib.sha512).digest()
    if not hmac.compare_digest(signature, expected_sig):
        raise ValueError("HMAC invalide : intégrité du fichier compromise !")

    key1 = master_key[:32]
    key2 = master_key[32:]

    iv1 = payload[:16]
    iv2 = payload[16:32]
    layer2_data = payload[32:]

    # Déchiffrement couche 2
    cipher2 = AES.new(key2, AES.MODE_CBC, iv2)
    layer1_data = unpad(cipher2.decrypt(layer2_data), AES.block_size)

    # Déchiffrement couche 1
    cipher1 = AES.new(key1, AES.MODE_CBC, iv1)
    original_data = unpad(cipher1.decrypt(layer1_data), AES.block_size)

    return original_data


# ── Upload vers Azure Blob Storage (remplace upload_to_s3) ────────────────────
def upload_to_azure_blob(payload: bytes, signature: bytes, container_name: str, blob_name: str) -> bool:
    """
    Envoie le fichier chiffré + sa signature vers Azure Blob Storage.
    
    Équivalent de l'ancienne fonction upload_to_s3().
    Le fichier final stocké = signature (64 octets) + payload.
    
    Args:
        payload       : données chiffrées
        signature     : signature HMAC-512 (64 octets)
        container_name: nom du conteneur Azure (ex: "documents-chiffres")
        blob_name     : chemin du fichier dans le conteneur (ex: "logs/user_123.bin")
    
    Returns:
        True si succès, False sinon
    """
    connection_string = os.environ.get(
        "AZURE_STORAGE_CONNECTION_STRING",
        ""
    )

    if not connection_string:
        print("[Blob Storage] AZURE_STORAGE_CONNECTION_STRING non définie.")
        return False

    try:
        # Assemblage : Signature (64 octets) + Payload chiffré
        body = signature + payload

        blob_service = BlobServiceClient.from_connection_string(connection_string)
        blob_client  = blob_service.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(body, overwrite=True)

        print(f"[Blob Storage] ✓ Fichier '{blob_name}' uploadé dans '{container_name}'")
        return True

    except Exception as e:
        print(f"[Blob Storage] Erreur upload: {e}")
        return False


# ── Alias pour compatibilité avec l'ancien code ───────────────────────────────
def upload_to_s3(payload: bytes, signature: bytes, bucket_name: str, s3_file_name: str) -> bool:
    """
    Alias de compatibilité.
    Redirige automatiquement vers Azure Blob Storage.
    'bucket_name' est ignoré (on utilise le conteneur 'documents-chiffres').
    """
    print("[Info] upload_to_s3() redirigé vers Azure Blob Storage.")
    container = "documents-chiffres"
    return upload_to_azure_blob(payload, signature, container, s3_file_name)
