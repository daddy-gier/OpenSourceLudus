#pragma once

#include "CoreMinimal.h"
#include "GameFramework/SaveGame.h"
#include "ViolationRecord.h"
#include "PunishmentRecord.h"
#include "PrisonSaveGame.generated.h"

USTRUCT()
struct FInmateSaveData
{
    GENERATED_BODY()

    UPROPERTY() FName InmateId;
    UPROPERTY() int32 AuthorityLevel;
    UPROPERTY() TArray<FViolationRecord> Violations;
    UPROPERTY() TArray<FPunishmentRecord> Punishments;
    UPROPERTY() FVector WorldLocation;
};

UCLASS()
class NYGHTSHADEHOLLOW_API UPrisonSaveGame : public USaveGame
{
    GENERATED_BODY()

public:
    UPROPERTY() float GameTimeSeconds;
    UPROPERTY() bool bLockdownActive;

    UPROPERTY() TArray<FInmateSaveData> Inmates;
};
