# Configuration du fournisseur AWS
provider "aws" {
  region = "us-east-1"
}

# 1. Création de la table DynamoDB pour remplacer ton dictionnaire TASKS
resource "aws_dynamodb_table" "tasks_table" {
  name           = "FaceAuthTasks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "task_id"

  attribute {
    name = "task_id"
    type = "S"
  }
}

# 2. Création du bucket S3 pour stocker tes fichiers et ton dataset
resource "aws_s3_bucket" "face_dataset" {
  # ATTENTION : Le nom du bucket doit être unique au monde. 
  # Ajoute ton nom ou des chiffres à la fin du nom ci-dessous :
  bucket = "pfe-face-auth-storage-votre-nom" 
}