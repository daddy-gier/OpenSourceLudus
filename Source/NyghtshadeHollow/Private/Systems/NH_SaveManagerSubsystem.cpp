#include "Systems/NH_SaveManagerSubsystem.h"
#include "Components/NH_AuthorityComponent.h"
#include "GameFramework/Actor.h"
#include "GameFramework/Character.h"
#include "Kismet/GameplayStatics.h"
#include "EngineUtils.h"

void UNH_SaveManagerSubsystem::StartAutosave()
{
    if (!GetWorld())
    {
        return;
    }

    GetWorld()->GetTimerManager().SetTimer(
        AutosaveHandle,
        this,
        &UNH_SaveManagerSubsystem::TriggerRollingAutosave,
        AutosaveIntervalSeconds,
        true
    );
}

void UNH_SaveManagerSubsystem::StopAutosave()
{
    if (GetWorld())
    {
        GetWorld()->GetTimerManager().ClearTimer(AutosaveHandle);
    }
}

void UNH_SaveManagerSubsystem::SavePrisonManual()
{
    SavePrisonToSlot(ManualSlotName);
}

void UNH_SaveManagerSubsystem::SavePrisonToSlot(const FString& SlotName)
{
    UNH_SaveGame* SaveData = GatherSaveData();
    if (!SaveData)
    {
        return;
    }

    UGameplayStatics::SaveGameToSlot(SaveData, SlotName, 0);
    OnPrisonSaved.Broadcast(SlotName);
}

bool UNH_SaveManagerSubsystem::LoadPrisonFromSlot(const FString& SlotName)
{
    if (!UGameplayStatics::DoesSaveGameExist(SlotName, 0))
    {
        return false;
    }

    UNH_SaveGame* SaveData = Cast<UNH_SaveGame>(UGameplayStatics::LoadGameFromSlot(SlotName, 0));
    if (!SaveData)
    {
        return false;
    }

    ApplySaveData(SaveData);
    OnPrisonLoaded.Broadcast(SlotName);
    return true;
}

void UNH_SaveManagerSubsystem::TriggerRollingAutosave()
{
    const FString SlotName = FString::Printf(TEXT("%s_%d"), *AutosaveSlotPrefix, AutosaveIndex);
    SavePrisonToSlot(SlotName);
    AutosaveIndex = (AutosaveIndex + 1) % FMath::Max(AutosaveSlots, 1);
}

void UNH_SaveManagerSubsystem::Deinitialize()
{
    StopAutosave();
    Super::Deinitialize();
}

UNH_SaveGame* UNH_SaveManagerSubsystem::GatherSaveData() const
{
    if (!GetWorld())
    {
        return nullptr;
    }

    UNH_SaveGame* SaveData = Cast<UNH_SaveGame>(UGameplayStatics::CreateSaveGameObject(UNH_SaveGame::StaticClass()));
    if (!SaveData)
    {
        return nullptr;
    }

    // TODO: Wire NH_GameState time + lockdown state when available.

    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        AActor* Actor = *It;
        if (!Actor)
        {
            continue;
        }

        if (ACharacter* Character = Cast<ACharacter>(Actor))
        {
            if (UNH_AuthorityComponent* Authority = Character->FindComponentByClass<UNH_AuthorityComponent>())
            {
                FNHSaveNPCData NPCData;
                NPCData.NPCName = Character->GetFName();
                NPCData.AuthorityLevel = Authority->CurrentAuthorityLevel;
                NPCData.Violations = Authority->ViolationHistory;
                NPCData.Punishments = Authority->ActivePunishments;
                SaveData->NPCs.Add(NPCData);
            }
        }
    }

    return SaveData;
}

void UNH_SaveManagerSubsystem::ApplySaveData(const UNH_SaveGame* SaveData) const
{
    if (!SaveData || !GetWorld())
    {
        return;
    }

    for (TActorIterator<ACharacter> It(GetWorld()); It; ++It)
    {
        ACharacter* Character = *It;
        if (!Character)
        {
            continue;
        }

        UNH_AuthorityComponent* Authority = Character->FindComponentByClass<UNH_AuthorityComponent>();
        if (!Authority)
        {
            continue;
        }

        const FNHSaveNPCData* FoundNPC = SaveData->NPCs.FindByPredicate([Character](const FNHSaveNPCData& Data)
        {
            return Data.NPCName == Character->GetFName();
        });

        if (!FoundNPC)
        {
            continue;
        }

        Authority->SetAuthorityLevel(FoundNPC->AuthorityLevel);
        Authority->ViolationHistory = FoundNPC->Violations;
        Authority->ActivePunishments = FoundNPC->Punishments;
    }
}
