#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "RevoltGPTBPLibrary.generated.h"

DECLARE_DYNAMIC_DELEGATE_TwoParams(FOnRevoltResponse, bool, bSuccess, const FString&, Response);

/**
 * Blueprint-callable helpers to interact with RevoltGPT.
 * These are intentionally minimal: they enqueue async requests and return via a dynamic delegate.
 */
UCLASS()
class URevoltGPTBPLibrary : public UBlueprintFunctionLibrary
{
  GENERATED_BODY()

public:
  // Generate text from a prompt. Callback receives success and raw response.
  UFUNCTION(BlueprintCallable, Category = "RevoltGPT")
  static void GenerateTextAsync(const FString& Prompt, const FOnRevoltResponse& Callback);

  // Request a blueprint implementation (server-side will return patch/diff or success)
  UFUNCTION(BlueprintCallable, Category = "RevoltGPT")
  static void ImplementBlueprintAsync(const FString& BlueprintName, const FString& BlueprintBody, const FOnRevoltResponse& Callback);
};
