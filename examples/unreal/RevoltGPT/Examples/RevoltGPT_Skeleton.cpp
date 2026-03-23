// RevoltGPT_Skeleton.cpp
// Single-file scaffold demonstrating RevoltGPT-style integration with Unreal Engine.
// Purpose: educational template for local AI agent integration (HTTP-based).
// NOTE: Replace API_URL and API_KEY with secure config. Do NOT hardcode in production.

#include "CoreMinimal.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"

// -----------------------------
// Config (replace with secure config, not hardcoded)
// -----------------------------
static const FString REVOLT_API_URL = TEXT("https://your-revoltgpt-server.example/api/v1");
static const FString REVOLT_API_KEY = TEXT("REPLACE_WITH_YOUR_KEY");

// -----------------------------
// Minimal RevoltGPT HTTP wrapper
// -----------------------------
namespace RevoltGPT
{
  // Callback signature
  using FOnResponse = TFunction<void(bool bSuccess, const FString& ResponseBody)>;

  // Helper: POST JSON to endpoint/<path>
  static void PostJson(const FString& Path, const FString& JsonPayload, FOnResponse Callback)
  {
    FString Url = REVOLT_API_URL / Path;
    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
    Request->SetURL(Url);
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->SetHeader(TEXT("Authorization"), FString::Printf(TEXT("Bearer %s"), *REVOLT_API_KEY));
    Request->SetContentAsString(JsonPayload);
    Request->OnProcessRequestComplete().BindLambda([Callback](FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bWasSuccessful)
    {
      if (!bWasSuccessful || !Resp.IsValid())
      {
        Callback(false, TEXT("HTTP request failed or invalid response"));
        return;
      }
      Callback(true, Resp->GetContentAsString());
    });
    Request->ProcessRequest();
  }

  // Simple synchronous-ish wrapper pattern using async callbacks:
  static void Initialize(const FString& Key)
  {
    // In a real system: store key securely or use secrets manager
    // Here we just log
    UE_LOG(LogTemp, Warning, TEXT("RevoltGPT::Initialize called (key placeholder)"));
  }

  static void Shutdown()
  {
    UE_LOG(LogTemp, Warning, TEXT("RevoltGPT::Shutdown called"));
  }

  // Parse file: send file content to agent for indexing/processing (async)
  static void ParseFile(const FString& RelativePath, const FString& Content, FOnResponse Callback = nullptr)
  {
    TSharedPtr<FJsonObject> Obj = MakeShareable(new FJsonObject);
    Obj->SetStringField(TEXT("event"), TEXT("parse_file"));
    Obj->SetStringField(TEXT("path"), RelativePath);
    Obj->SetStringField(TEXT("content"), Content);

    FString Payload;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Payload);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);
    PostJson(TEXT("parse"), Payload, [Callback](bool bOk, const FString& Resp)
    {
      if (Callback)
      {
        Callback(bOk, Resp);
      }
      if (bOk)
      {
        UE_LOG(LogTemp, Log, TEXT("ParseFile response: %s"), *Resp);
      }
    });
  }

  // Generate text from a prompt
  static void GenerateText(const FString& Prompt, FOnResponse Callback)
  {
    TSharedPtr<FJsonObject> Obj = MakeShareable(new FJsonObject);
    Obj->SetStringField(TEXT("prompt"), Prompt);
    Obj->SetNumberField(TEXT("max_tokens"), 512);

    FString Payload;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Payload);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);
    PostJson(TEXT("generate"), Payload, Callback);
  }

  // Implement blueprint / create patch request (server does heavy-lifting)
  static void ImplementBlueprint(const FString& BlueprintName, const FString& BlueprintText, FOnResponse Callback = nullptr)
  {
    TSharedPtr<FJsonObject> Obj = MakeShareable(new FJsonObject);
    Obj->SetStringField(TEXT("action"), TEXT("implement_blueprint"));
    Obj->SetStringField(TEXT("blueprint_name"), BlueprintName);
    Obj->SetStringField(TEXT("blueprint_text"), BlueprintText);

    FString Payload;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Payload);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);
    PostJson(TEXT("implement"), Payload, Callback);
  }

  // Generic analyze game state
  static void AnalyzeGameState(const FString& StateJson, FOnResponse Callback)
  {
    TSharedPtr<FJsonObject> Obj = MakeShareable(new FJsonObject);
    Obj->SetStringField(TEXT("action"), TEXT("analyze_state"));
    Obj->SetStringField(TEXT("state_json"), StateJson);

    FString Payload;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Payload);
    FJsonSerializer::Serialize(Obj.ToSharedRef(), Writer);
    PostJson(TEXT("analyze"), Payload, Callback);
  }
}

// -----------------------------
// RevoltGPT Module (Plugin-like)
// -----------------------------
class FRevoltGPTModule : public IModuleInterface
{
public:
  virtual void StartupModule() override
  {
    RevoltGPT::Initialize(REVOLT_API_KEY);
    UE_LOG(LogTemp, Warning, TEXT("RevoltGPT module startup - scanning project..."));
    ScanProjectFilesAsync();
  }

  virtual void ShutdownModule() override
  {
    RevoltGPT::Shutdown();
  }

private:
  // Asynchronous project scan (fires off ParseFile calls)
  void ScanProjectFilesAsync()
  {
    // Note: scanning Content dir - in large projects optimize and chunk
    FString ContentDir = FPaths::ProjectContentDir();
    TArray<FString> Files;
    IFileManager::Get().FindFilesRecursively(Files, *ContentDir, TEXT("*.*"));

    for (const FString& FullPath : Files)
    {
      FString FileContent;
      if (FFileHelper::LoadFileToString(FileContent, *FullPath))
      {
        // Send relative path for better server indexing
        FString RelPath = FullPath;
        FPaths::MakePathRelativeTo(RelPath, *FPaths::ProjectDir());
        RevoltGPT::ParseFile(RelPath, FileContent, nullptr);
      }
    }
    UE_LOG(LogTemp, Warning, TEXT("Project scan requests queued (%d files)"), Files.Num());
  }
};

IMPLEMENT_MODULE(FRevoltGPTModule, RevoltGPTPlugin)

// -----------------------------
// Utility: Simple JSON helper to extract "text" from model response
// -----------------------------
static bool ExtractGeneratedText(const FString& JsonStr, FString& OutText)
{
  TSharedPtr<FJsonObject> Root;
  TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonStr);
  if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
  {
    return false;
  }
  if (Root->HasField(TEXT("text")))
  {
    OutText = Root->GetStringField(TEXT("text"));
    return true;
  }
  // Fallback to data[0].text
  if (Root->HasTypedField<EJson::Array>(TEXT("data")))
  {
    const TArray<TSharedPtr<FJsonValue>> Arr = Root->GetArrayField(TEXT("data"));
    if (Arr.Num() && Arr[0].IsValid() && Arr[0]->AsObject()->HasField(TEXT("text")))
    {
      OutText = Arr[0]->AsObject()->GetStringField(TEXT("text"));
      return true;
    }
  }
  return false;
}

// -----------------------------
// AI Blueprint Manager (invokable from Blueprints/C++ tooling)
// -----------------------------
class FAI_BlueprintManager
{
public:
  // Create a simple enemy AI blueprint logic via RevoltGPT
  static void CreateEnemyAI(const FString& BlueprintName)
  {
    const FString BlueprintLogic = TEXT(
      "Event Tick:\n"
      " if PlayerInSight():\n"
      "  MoveTo(PlayerLocation)\n"
      " else:\n"
      "  PatrolRoute()\n"
    );

    RevoltGPT::ImplementBlueprint(BlueprintName, BlueprintLogic, [](bool bOk, const FString& Resp)
    {
      if (bOk)
      {
        UE_LOG(LogTemp, Warning, TEXT("CreateEnemyAI ok: %s"), *Resp);
      }
      else
      {
        UE_LOG(LogTemp, Error, TEXT("CreateEnemyAI failed: %s"), *Resp);
      }
    });
  }

  // Update NPC dialogue in-game by generating text given a prompt
  static void UpdateNPCDialogue(AActor* WorldContextActor, const FString& NPCName, const FString& Prompt)
  {
    RevoltGPT::GenerateText(Prompt, [WorldContextActor, NPCName](bool bOk, const FString& Resp)
    {
      if (!bOk)
      {
        UE_LOG(LogTemp, Error, TEXT("GenerateText failed: %s"), *Resp);
        return;
      }

      FString Generated;
      if (!ExtractGeneratedText(Resp, Generated))
      {
        Generated = Resp;
      }

      // Find NPC actor by name and set a hypothetical property "Dialogue"
      if (WorldContextActor)
      {
        UWorld* W = WorldContextActor->GetWorld();
        if (!W)
        {
          return;
        }
        for (TActorIterator<AActor> It(W); It; ++It)
        {
          AActor* A = *It;
          if (A && A->GetName().Contains(NPCName))
          {
            // This is demo code. Replace with your NPC interface
            UE_LOG(LogTemp, Warning, TEXT("Would set dialogue on %s -> %s"), *A->GetName(), *Generated);
            // Example: Cast to ANPCCharacter* and call SetDialogue(Generated)
          }
        }
      }
    });
  }
};

// -----------------------------
// GameState Analyzer (sends JSON snapshot to agent)
// -----------------------------
class FGameStateAnalyzer
{
public:
  static void AnalyzeAndOptimize(UWorld* World)
  {
    if (!World)
    {
      return;
    }

    // Build a tiny JSON snapshot - extend with whatever you need
    TSharedPtr<FJsonObject> Root = MakeShareable(new FJsonObject);
    Root->SetStringField(TEXT("level"), World->GetMapName());
    Root->SetNumberField(TEXT("actor_count"), World->GetActorCount());
    // ... add more game-specific metrics: player stats, spawn counts, etc.

    FString OutJson;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutJson);
    FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);

    RevoltGPT::AnalyzeGameState(OutJson, [](bool bOk, const FString& Resp)
    {
      if (bOk)
      {
        UE_LOG(LogTemp, Warning, TEXT("AnalyzeGameState response: %s"), *Resp);
      }
      else
      {
        UE_LOG(LogTemp, Error, TEXT("AnalyzeGameState error: %s"), *Resp);
      }
    });
  }
};

// -----------------------------
// Player Interaction (example actor method you can call)
// -----------------------------
class ARevoltPlayerInteractionActor : public AActor
{
public:
  ARevoltPlayerInteractionActor()
  {
    PrimaryActorTick.bCanEverTick = false;
  }

  // Example: when player talks to NPC, generate context-aware dialogue
  UFUNCTION(BlueprintCallable, Category = "RevoltGPT")
  void TalkToNPC(const FString& NPCName)
  {
    // Collect simple player context - in a real project add inventory, stats, quests
    FString PlayerSummary = TEXT("player_level:10; health:85; items:sword,apple");

    FString Prompt = FString::Printf(TEXT("Generate a friendly greeting for an NPC named %s based on player state: %s"), *NPCName, *PlayerSummary);

    RevoltGPT::GenerateText(Prompt, [this, NPCName](bool bOk, const FString& Resp)
    {
      if (!bOk)
      {
        UE_LOG(LogTemp, Error, TEXT("TalkToNPC failed: %s"), *Resp);
        return;
      }
      FString Generated;
      if (!ExtractGeneratedText(Resp, Generated))
      {
        Generated = Resp;
      }
      UE_LOG(LogTemp, Warning, TEXT("TalkToNPC -> NPC(%s) dialogue: %s"), *NPCName, *Generated);

      // Here you'd find the NPC actor and set its dialogue or trigger a UI element
    });
  }
};

// -----------------------------
// Asset Generator (request server to create procedural asset code / commands)
// -----------------------------
class FAssetGenerator
{
public:
  static void GenerateEnvironmentAsset(const FString& AssetPrompt)
  {
    RevoltGPT::GenerateText(AssetPrompt, [](bool bOk, const FString& Resp)
    {
      if (!bOk)
      {
        UE_LOG(LogTemp, Error, TEXT("GenerateEnvironmentAsset failed: %s"), *Resp);
        return;
      }
      FString Generated;
      if (!ExtractGeneratedText(Resp, Generated))
      {
        Generated = Resp;
      }
      UE_LOG(LogTemp, Warning, TEXT("Generated asset script or description:\n%s"), *Generated);

      // Optionally: send this back to server endpoint to create uasset or to a local tool to convert into actual assets.
    });
  }
};

// -----------------------------
// Example usage (could be called from editor utility or Blueprint)
// -----------------------------
// FAI_BlueprintManager::CreateEnemyAI(TEXT("BP_EnemyAI_Custom"));
// FGameStateAnalyzer::AnalyzeAndOptimize(GetWorld());
// FAssetGenerator::GenerateEnvironmentAsset(TEXT("Create a ruined tower mesh with top walkway and ivy"));
// (Or call ARevoltPlayerInteractionActor->TalkToNPC from Blueprint)
