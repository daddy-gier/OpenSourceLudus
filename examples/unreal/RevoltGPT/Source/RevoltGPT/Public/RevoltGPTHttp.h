#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "Http.h"
#include "Templates/SharedPointer.h"

DECLARE_DELEGATE_TwoParams(FRevoltHttpResponseDelegate, bool /*bSuccess*/, const FString& /*ResponseBody*/);

/**
 * Minimal RevoltGPT HTTP wrapper.
 * Replace API_URL and use secure key handling.
 */
namespace RevoltHttp
{
  // NOTE: Do not hardcode secrets in source. This is for scaffold only.
  static const FString API_URL = TEXT("https://your-revoltgpt-server.example/api/v1");

  // POST JSON to a path (e.g. "generate", "parse")
  static void PostJson(const FString& Path, const FString& JsonPayload, FRevoltHttpResponseDelegate Callback);
}
