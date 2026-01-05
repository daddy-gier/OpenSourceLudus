#include "RevoltGPTModule.h"

#include "Interfaces/IPluginManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#define LOCTEXT_NAMESPACE "FRevoltGPTModule"

void FRevoltGPTModule::StartupModule()
{
  UE_LOG(LogTemp, Display, TEXT("[RevoltGPT] StartupModule"));

  // Example: read local config (recommended: env or secrets manager instead)
  FString PluginDir = GetPluginBaseDir();
  FString ConfigPath = FPaths::Combine(PluginDir, TEXT("Config/revoltgpt.conf"));
  FString ConfigContent;
  if (FPaths::FileExists(ConfigPath) && FFileHelper::LoadFileToString(ConfigContent, *ConfigPath))
  {
    UE_LOG(LogTemp, Display, TEXT("[RevoltGPT] Found config at %s"), *ConfigPath);
    // parse if needed; sample shows where you could load non-secret defaults
  }
  else
  {
    UE_LOG(LogTemp, Warning, TEXT("[RevoltGPT] No local config found. Ensure secrets stored securely."));
  }

  // Optional: kick off project scan in background (be careful in large projects)
  // FRevoltGPTHelper::StartProjectScan(); // example hook, not implemented here
}

void FRevoltGPTModule::ShutdownModule()
{
  UE_LOG(LogTemp, Display, TEXT("[RevoltGPT] ShutdownModule"));
}

FString FRevoltGPTModule::GetPluginBaseDir()
{
  TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("RevoltGPT"));
  if (Plugin.IsValid())
  {
    return Plugin->GetBaseDir();
  }
  return FString();
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FRevoltGPTModule, RevoltGPT)
