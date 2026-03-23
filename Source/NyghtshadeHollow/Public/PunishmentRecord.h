#pragma once

#include "CoreMinimal.h"
#include "PunishmentTypes.h"
#include "PunishmentRecord.generated.h"

USTRUCT(BlueprintType)
struct FPunishmentRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    EPunishmentType Type;

    UPROPERTY(BlueprintReadOnly)
    float StartGameTime;

    UPROPERTY(BlueprintReadOnly)
    float Duration;

    UPROPERTY(BlueprintReadOnly)
    float TimeServed;

    UPROPERTY(BlueprintReadOnly)
    bool bEligibleForReview;

    UPROPERTY(BlueprintReadOnly)
    FString Reason;
};
