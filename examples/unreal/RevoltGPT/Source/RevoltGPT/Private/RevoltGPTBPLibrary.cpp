#include "RevoltGPTBPLibrary.h"

#include "Dom/JsonObject.h"
#include "RevoltGPTHttp.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

void URevoltGPTBPLibrary::GenerateTextAsync(const FString& Prompt, const FOnRevoltResponse& Callback)
{
  TSharedPtr<FJsonObject> Root = MakeShareable(new FJsonObject);
  Root->SetStringField(TEXT("prompt"), Prompt);
  Root->SetNumberField(TEXT("max_tokens"), 512);

  FString Payload;
  TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Payload);
  FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);

  RevoltHttp::PostJson(TEXT("generate"), Payload, FRevoltHttpResponseDelegate::CreateLambda([Callback](bool bOk, const FString& Resp)
  {
    // Forward raw server response to Blueprint callback
    Callback.ExecuteIfBound(bOk, Resp);
  }));
}

void URevoltGPTBPLibrary::ImplementBlueprintAsync(const FString& BlueprintName, const FString& BlueprintBody, const FOnRevoltResponse& Callback)
{
  TSharedPtr<FJsonObject> Root = MakeShareable(new FJsonObject);
  Root->SetStringField(TEXT("action"), TEXT("implement_blueprint"));
  Root->SetStringField(TEXT("blueprint_name"), BlueprintName);
  Root->SetStringField(TEXT("blueprint_body"), BlueprintBody);

  FString Payload;
  TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Payload);
  FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);

  RevoltHttp::PostJson(TEXT("implement"), Payload, FRevoltHttpResponseDelegate::CreateLambda([Callback](bool bOk, const FString& Resp)
  {
    Callback.ExecuteIfBound(bOk, Resp);
  }));
}
