import hashlib
import hmac
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import qrcode
import io
import uuid
import boto3
import random
import string

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
    """Génère un code aléatoire et un QR code contenant l'URL de validation."""
    
    # 1. Générer un code aléatoire format TR-XXX-XXX
    part1 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    part2 = ''.join(random.choices(string.digits, k=3))
    random_trust_code = f"TR-{part1}-{part2}"

    # 2. Construire l'URL avec le code en paramètre pour la Lambda
    api_url = "https://z9v2z7b41e.execute-api.us-east-1.amazonaws.com/default/VerifyMFATask"
    url_validation = f"{api_url}?task_id={task_id}&code={random_trust_code}"
    
    # 3. Générer l'image du QR Code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url_validation)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    qr_bytes = img_buffer.getvalue()
    
    # 4. (Optionnel) Upload vers S3 pour garder une trace
    try:
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket_name, Key=f"qr_codes/{user_name}_auth_qr.png", Body=qr_bytes, ContentType='image/png')
    except: pass
    
    # On retourne le code (pour le PC) et l'image (pour l'affichage)
    return random_trust_code, qr_bytes