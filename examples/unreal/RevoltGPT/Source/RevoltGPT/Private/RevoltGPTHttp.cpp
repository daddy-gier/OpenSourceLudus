#include "RevoltGPTHttp.h"

#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "RevoltGPTSecure.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

void RevoltHttp::PostJson(const FString& Path, const FString& JsonPayload, FRevoltHttpResponseDelegate Callback)
{
  RevoltSecure::InitApiKey();

  FString Url = API_URL / Path;
  TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
  Request->SetURL(Url);
  Request->SetVerb(TEXT("POST"));
  Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));

  const FString& ApiKey = RevoltSecure::GetApiKey();
  if (ApiKey.IsEmpty())
  {
    UE_LOG(LogTemp, Error, TEXT("[RevoltHttp] No API key. Request will likely fail."));
  }
  Request->SetHeader(TEXT("Authorization"), FString::Printf(TEXT("Bearer %s"), *ApiKey));
  Request->SetContentAsString(JsonPayload);

  Request->OnProcessRequestComplete().BindLambda([Callback](FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bWasSuccessful)
  {
    if (!bWasSuccessful || !Resp.IsValid())
    {
      Callback.ExecuteIfBound(false, TEXT("HTTP failed or invalid response"));
      return;
    }
    Callback.ExecuteIfBound(true, Resp->GetContentAsString());
  });

  Request->ProcessRequest();
}
