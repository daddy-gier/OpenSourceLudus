#pragma once

#include "CoreMinimal.h"
#include "ViolationTypes.h"
#include "ViolationRecord.generated.h"

USTRUCT(BlueprintType)
struct FViolationRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    EViolationType Type;

    UPROPERTY(BlueprintReadOnly)
    int32 Severity;

    UPROPERTY(BlueprintReadOnly)
    float GameTimeStamp;

    UPROPERTY(BlueprintReadOnly)
    FName ReportingGuard;

    UPROPERTY(BlueprintReadOnly)
    FString Context;
};
