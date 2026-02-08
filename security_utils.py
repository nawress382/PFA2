import hashlib
import hmac
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

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