from security_utils import encrypt_aes_512, upload_to_s3

# 1. On prépare une donnée (ça peut être une image ou un texte)
ma_donnee = b"Ceci est une image de visage tres confidentielle."

# 2. On chiffre en double AES (512 bits total)
print("Chiffrement en cours...")
payload, signature = encrypt_aes_512(ma_donnee)

# 3. On envoie sur ton Bucket S3 AWS
# REMPLACE PAR TON NOM DE BUCKET TERRAFORM :
MON_BUCKET = "pfe-face-auth-storage-votre-nom" 

print("Envoi vers AWS S3...")
success = upload_to_s3(payload, signature, MON_BUCKET, "visage_chiffre.bin")

if success:
    print("MIGRATION RÉUSSIE : Ton fichier est sur le Cloud et il est illisible sans la clé !")