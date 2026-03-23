#include "RevoltGPTSecure.h"

#include "HAL/PlatformMisc.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace RevoltSecure
{
  static FString CachedApiKey = TEXT("");

  static FString ReadApiKeyFromEnv()
  {
    // Common name - set this on your system before launching UnrealEditor
    return FPlatformMisc::GetEnvironmentVariable(TEXT("REVOLT_API_KEY"));
  }

  static FString ReadApiKeyFromConfigFile()
  {
    TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("RevoltGPT"));
    if (!Plugin.IsValid())
    {
      return TEXT("");
    }

    FString ConfPath = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Config/revoltgpt.conf"));

    if (!FPaths::FileExists(ConfPath))
    {
      return TEXT("");
    }

    FString FileContent;
    if (!FFileHelper::LoadFileToString(FileContent, *ConfPath))
    {
      return TEXT("");
    }

    // crude parsing for ApiKey= line
    TArray<FString> Lines;
    FileContent.ParseIntoArrayLines(Lines);
    for (const FString& Line : Lines)
    {
      FString Trimmed = Line.TrimStartAndEnd();
      if (Trimmed.StartsWith(TEXT("ApiKey=")))
      {
        FString Value = Trimmed.RightChop(7); // length of "ApiKey="
        return Value.TrimStartAndEnd();
      }
    }
    return TEXT("");
  }

  void InitApiKey()
  {
    if (!CachedApiKey.IsEmpty())
    {
      return;
    }

    FString Key = ReadApiKeyFromEnv();
    if (!Key.IsEmpty())
    {
      CachedApiKey = Key;
      UE_LOG(LogTemp, Display, TEXT("[RevoltSecure] Loaded API key from REVOLT_API_KEY env var"));
      return;
    }

    // fallback (only if env var missing)
    Key = ReadApiKeyFromConfigFile();
    if (!Key.IsEmpty())
    {
      CachedApiKey = Key;
      UE_LOG(LogTemp, Warning, TEXT("[RevoltSecure] Loaded API key from local config (NOT recommended)"));
      return;
    }

    UE_LOG(LogTemp, Error, TEXT("[RevoltSecure] No Revolt API key found. Please set REVOLT_API_KEY env var"));
  }

  const FString& GetApiKey()
  {
    return CachedApiKey;
  }
}
