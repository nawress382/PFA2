output "resource_group_name" {
  description = "Nom du groupe de ressources"
  value       = azurerm_resource_group.pfa2_rg.name
}

output "storage_account_name" {
  description = "Nom du compte de stockage"
  value       = azurerm_storage_account.pfa2_storage.name
}

output "cosmos_endpoint" {
  description = "Endpoint Cosmos DB"
  value       = azurerm_cosmosdb_account.pfa2_cosmos.endpoint
}

output "keyvault_uri" {
  description = "URI du Key Vault"
  value       = azurerm_key_vault.pfa2_kv.vault_uri
}

output "app_service_url" {
  description = "URL de votre API"
  value       = "https://${azurerm_linux_web_app.pfa2_api.default_hostname}"
}