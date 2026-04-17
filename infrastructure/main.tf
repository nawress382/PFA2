# ── PROVIDER : on dit à Terraform qu'on utilise Azure ──────────────────
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }
  required_version = ">= 1.5.0"
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
}

# ── 1. GROUPE DE RESSOURCES ─────────────────────────────────────────────
resource "azurerm_resource_group" "pfa2_rg" {
  name     = var.resource_group_name
  location = var.location
}

# ── 2. STOCKAGE BLOB (remplace S3) ─────────────────────────────────────
resource "azurerm_storage_account" "pfa2_storage" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.pfa2_rg.name
  location                 = azurerm_resource_group.pfa2_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  # Chiffrement activé par défaut
  blob_properties {
    versioning_enabled = true
  }

  tags = {
    projet = "PFA2"
    auteur = "Marzouki Nawress"
  }
}

# Conteneur pour les documents chiffrés
resource "azurerm_storage_container" "documents_chiffres" {
  name                  = "documents-chiffres"
  storage_account_name  = azurerm_storage_account.pfa2_storage.name
  container_access_type = "private"
}

# Conteneur pour les logs d'audit
resource "azurerm_storage_container" "logs_audit" {
  name                  = "logs-audit"
  storage_account_name  = azurerm_storage_account.pfa2_storage.name
  container_access_type = "private"
}

# ── 3. COSMOS DB (remplace DynamoDB) ───────────────────────────────────
resource "azurerm_cosmosdb_account" "pfa2_cosmos" {
  name                = var.cosmos_account_name
  location            = azurerm_resource_group.pfa2_rg.location
  resource_group_name = azurerm_resource_group.pfa2_rg.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  # Mode Serverless = gratuit pour petits volumes
  capabilities {
    name = "EnableServerless"
  }

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = azurerm_resource_group.pfa2_rg.location
    failover_priority = 0
  }

  tags = {
    projet = "PFA2"
  }
}

# Base de données
resource "azurerm_cosmosdb_sql_database" "face_auth_db" {
  name                = "FaceAuthDB"
  resource_group_name = azurerm_resource_group.pfa2_rg.name
  account_name        = azurerm_cosmosdb_account.pfa2_cosmos.name
}

# Conteneur (équivalent table DynamoDB)
resource "azurerm_cosmosdb_sql_container" "auth_tasks" {
  name                = "AuthTasks"
  resource_group_name = azurerm_resource_group.pfa2_rg.name
  account_name        = azurerm_cosmosdb_account.pfa2_cosmos.name
  database_name       = azurerm_cosmosdb_sql_database.face_auth_db.name
  partition_key_paths = ["/user_id"]
}

# ── 4. KEY VAULT ────────────────────────────────────────────────────────
data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "pfa2_kv" {
  name                = var.keyvault_name
  location            = azurerm_resource_group.pfa2_rg.location
  resource_group_name = azurerm_resource_group.pfa2_rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge"
    ]
  }
}

# Stocker la clé AES dans Key Vault
resource "azurerm_key_vault_secret" "aes_key" {
  name         = "AES-SECRET-KEY"
  value        = var.aes_secret_key
  key_vault_id = azurerm_key_vault.pfa2_kv.id
}

# Stocker la clé HMAC dans Key Vault
resource "azurerm_key_vault_secret" "hmac_key" {
  name         = "HMAC-SECRET-KEY"
  value        = var.hmac_secret_key
  key_vault_id = azurerm_key_vault.pfa2_kv.id
}

# Stocker la clé Cosmos DB dans Key Vault
resource "azurerm_key_vault_secret" "cosmos_key" {
  name         = "COSMOS-PRIMARY-KEY"
  value        = azurerm_cosmosdb_account.pfa2_cosmos.primary_key
  key_vault_id = azurerm_key_vault.pfa2_kv.id
}

# Stocker la chaîne de connexion Storage dans Key Vault
resource "azurerm_key_vault_secret" "storage_connection" {
  name         = "STORAGE-CONNECTION-STRING"
  value        = azurerm_storage_account.pfa2_storage.primary_connection_string
  key_vault_id = azurerm_key_vault.pfa2_kv.id
}

# ── 5. APP SERVICE (votre api_server.py) ────────────────────────────────
resource "azurerm_service_plan" "pfa2_plan" {
  name                = "pfa2-service-plan"
  location            = azurerm_resource_group.pfa2_rg.location
  resource_group_name = azurerm_resource_group.pfa2_rg.name
  os_type             = "Linux"
  sku_name            = "F1"  # Gratuit
}

resource "azurerm_linux_web_app" "pfa2_api" {
  name                = var.app_service_name
  location            = azurerm_resource_group.pfa2_rg.location
  resource_group_name = azurerm_resource_group.pfa2_rg.name
  service_plan_id     = azurerm_service_plan.pfa2_plan.id

  site_config {
  always_on = false 
    application_stack {
      python_version = "3.11"
    }
  }

  app_settings = {
    "AZURE_STORAGE_CONNECTION_STRING" = azurerm_storage_account.pfa2_storage.primary_connection_string
    "COSMOS_ENDPOINT"                 = azurerm_cosmosdb_account.pfa2_cosmos.endpoint
    "COSMOS_KEY"                      = azurerm_cosmosdb_account.pfa2_cosmos.primary_key
    "KEYVAULT_URI"                    = azurerm_key_vault.pfa2_kv.vault_uri
  }

  tags = {
    projet = "PFA2"
  }
}