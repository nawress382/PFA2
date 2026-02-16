import hashlib
import hmac
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import qrcode
import io
import uuid
import boto3

# Utilisons une méthode plus sûre pour générer une clé de 512 bits (64 octets)
# à partir de n'importe quel texte :
MON_TEXTE_SECRET = "Ceci est mon mot de passe secret pour le PFE" 
# On transforme ce texte en une clé de 64 octets exacte grâce au hachage SHA-512
MASTER_KEY_512 = hashlib.sha512(MON_TEXTE_SECRET.encode()).digest()

def encrypt_aes_512(data: bytes):
    # Maintenant MASTER_KEY_512 fait exactement 64 octets !
    key1 = MASTER_KEY_512[:32] # 32 octets (256 bits)
    key2 = MASTER_KEY_512[32:] # 32 octets (256 bits)

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
    signature = hmac.new(MASTER_KEY_512, final_payload, hashlib.sha512).digest()

    return final_payload, signature

def upload_to_s3(payload, signature, bucket_name, s3_file_name):
    """Envoie le fichier chiffré et sa signature sur S3."""
    import boto3
    s3 = boto3.client('s3')
    
    # On assemble le fichier final (Signature + Données)
    # Comme ça, le fichier sur le cloud est totalement protégé
    body = signature + payload
    
    try:
        s3.put_object(Bucket=bucket_name, Key=s3_file_name, Body=body)
        return True
    except Exception as e:
        print(f"Erreur S3: {e}")
        return False
def generate_and_upload_qr(user_name, bucket_name, task_id):
    """Génère un QR code contenant un lien de validation mobile, l'envoie sur S3."""
    

    # 1. TON URL API GATEWAY (Remplace par celle que tu as copiée sur AWS)
    # Exemple : https://abcd123.execute-api.us-east-1.amazonaws.com/default/VerifyMFATask
    api_url = " https://z9v2z7b41e.execute-api.us-east-1.amazonaws.com/default/VerifyMFATask"
    
    # On construit le lien de validation final
    url_validation = f"{api_url}?task_id={task_id}"
    
    # 2. Créer une signature de secours (Trust Code)
    trust_code = f"TRUST-{user_name}-{uuid.uuid4().hex[:4].upper()}"

    # 3. Générer l'image du QR Code avec l'URL de validation
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url_validation) # <-- C'est l'URL que le téléphone va scanner
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # 4. Sauvegarder l'image dans un buffer
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    qr_bytes = img_buffer.getvalue()
    img_buffer.seek(0)
    
    # 5. Envoi vers S3
    try:
        s3 = boto3.client('s3')
        s3_path = f"qr_codes/{user_name}_auth_qr.png"
        s3.put_object(
            Bucket=bucket_name, 
            Key=s3_path, 
            Body=img_buffer, 
            ContentType='image/png'
        )
        print(f"DEBUG: QR Code (URL) envoyé sur S3 avec succès.")
    except Exception as e:
        print(f"Erreur S3 QR Code: {e}")
    
    # On retourne l'URL (pour info) et les bytes de l'image (pour l'affichage PyQt6)
    return url_validation, qr_bytes