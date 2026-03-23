#pragma once

#include "CoreMinimal.h"
#include "GameFramework/SaveGame.h"
#include "Authority/NH_AuthorityTypes.h"
#include "NH_SaveGame.generated.h"

USTRUCT(BlueprintType)
struct FNHSaveDoorData
{
    GENERATED_BODY()

    UPROPERTY()
    FName DoorName;

    UPROPERTY()
    bool bIsLocked = false;

    UPROPERTY()
    FString OverrideReason;
};

USTRUCT(BlueprintType)
struct FNHSaveNPCData
{
    GENERATED_BODY()

    UPROPERTY()
    FName NPCName;

    UPROPERTY()
    FName CurrentActivity;

    UPROPERTY()
    FName CurrentTargetPoint;

    UPROPERTY()
    ENHAuthorityLevel AuthorityLevel = ENHAuthorityLevel::Free;

    UPROPERTY()
    TArray<FNHViolationRecord> Violations;

    UPROPERTY()
    TArray<FNHPunishmentRecord> Punishments;
};

UCLASS()
class NYGHTSHADEHOLLOW_API UNH_SaveGame : public USaveGame
{
    GENERATED_BODY()

public:
    UPROPERTY()
    int32 Hour = 0;

    UPROPERTY()
    int32 Minute = 0;

    UPROPERTY()
    bool bLockdownActive = false;

    UPROPERTY()
    FString LockdownCause;

    UPROPERTY()
    TArray<FNHSaveDoorData> Doors;

    UPROPERTY()
    TArray<FNHSaveNPCData> NPCs;
};
