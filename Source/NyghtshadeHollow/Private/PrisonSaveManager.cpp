#include "PrisonSaveManager.h"
#include "PrisonSaveGame.h"
#include "NH_GameState.h"
#include "InmateAuthorityComponent.h"
#include "InmateJudicialComponent.h"
#include "Kismet/GameplayStatics.h"
#include "EngineUtils.h"
#include "GameFramework/Character.h"

APrisonSaveManager::APrisonSaveManager()
{
    PrimaryActorTick.bCanEverTick = false;
}

void APrisonSaveManager::SavePrison()
{
    UPrisonSaveGame* Save = Cast<UPrisonSaveGame>(
        UGameplayStatics::CreateSaveGameObject(UPrisonSaveGame::StaticClass())
    );

    if (!Save)
    {
        return;
    }

    auto* GS = GetWorld() ? GetWorld()->GetGameState<ANH_GameState>() : nullptr;
    Save->GameTimeSeconds = GS ? GS->GetServerWorldTimeSeconds() : 0.f;

    for (TActorIterator<ACharacter> It(GetWorld()); It; ++It)
    {
        auto* Authority = It->FindComponentByClass<UInmateAuthorityComponent>();
        auto* Judicial = It->FindComponentByClass<UInmateJudicialComponent>();
        if (!Authority || !Judicial)
        {
            continue;
        }

        FInmateSaveData Data;
        Data.InmateId = It->GetFName();
        Data.AuthorityLevel = Authority->AuthorityLevel;
        Data.Violations = Authority->ViolationHistory;
        Data.Punishments = Judicial->ActivePunishments;
        Data.WorldLocation = It->GetActorLocation();

        Save->Inmates.Add(Data);
    }

    UGameplayStatics::SaveGameToSlot(Save, SaveSlot, 0);
    OnPrisonSaved.Broadcast();
}

void APrisonSaveManager::LoadPrison()
{
    auto* Save = Cast<UPrisonSaveGame>(
        UGameplayStatics::LoadGameFromSlot(SaveSlot, 0)
    );
    if (!Save)
    {
        return;
    }

    for (TActorIterator<ACharacter> It(GetWorld()); It; ++It)
    {
        for (const auto& Data : Save->Inmates)
        {
            if (It->GetFName() != Data.InmateId)
            {
                continue;
            }

            auto* Authority = It->FindComponentByClass<UInmateAuthorityComponent>();
            auto* Judicial = It->FindComponentByClass<UInmateJudicialComponent>();
            if (!Authority || !Judicial)
            {
                continue;
            }

            Authority->AuthorityLevel = Data.AuthorityLevel;
            Authority->ViolationHistory = Data.Violations;
            Judicial->ActivePunishments = Data.Punishments;

            It->SetActorLocation(Data.WorldLocation);
        }
    }
}
