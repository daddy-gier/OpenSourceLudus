#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FRevoltGPTModule : public IModuleInterface
{
public:
  virtual void StartupModule() override;
  virtual void ShutdownModule() override;

  // Helper: get plugin dir for config files
  static FString GetPluginBaseDir();
};
