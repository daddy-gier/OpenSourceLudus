#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "RevoltGPTSecureBPLibrary.generated.h"

UCLASS()
class URevoltGPTSecureBPLibrary : public UBlueprintFunctionLibrary
{
  GENERATED_BODY()

public:
  UFUNCTION(BlueprintCallable, Category = "RevoltGPT|Secure")
  static void InitRevoltApiKeyFromEnv();
};
