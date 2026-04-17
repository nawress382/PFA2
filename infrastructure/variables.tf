variable "location" {
  description = "Région Azure"
  type        = string
  default     = "France Central"
}

variable "resource_group_name" {
  description = "Nom du groupe de ressources"
  type        = string
  default     = "PFA2-SecurityCloud-RG"
}

variable "storage_account_name" {
  description = "Nom du compte de stockage (minuscules uniquement)"
  type        = string
  default     = "pfa2storageauth"
}

variable "cosmos_account_name" {
  description = "Nom du compte Cosmos DB"
  type        = string
  default     = "pfa2-cosmos-auth"
}

variable "keyvault_name" {
  description = "Nom du Key Vault"
  type        = string
  default     = "pfa2-keyvault-auth"
}

variable "app_service_name" {
  description = "Nom de l'App Service"
  type        = string
  default     = "pfa2-api-faceauth"
}

variable "aes_secret_key" {
  description = "Clé secrète AES-256"
  type        = string
  sensitive   = true
}

variable "hmac_secret_key" {
  description = "Clé secrète HMAC-512"
  type        = string
  sensitive   = true
}