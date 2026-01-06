#pragma once

#include "CoreMinimal.h"
#include "Engine/DataTable.h"
#include "ScheduleRow.generated.h"

USTRUCT(BlueprintType)
struct FScheduleRow : public FTableRowBase
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    FName NPCId;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    int32 StartHour = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    int32 StartMinute = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    int32 EndHour = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    int32 EndMinute = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    FName TargetPointName;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    FName ActivityName;
};
