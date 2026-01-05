#pragma once

#include "CoreMinimal.h"
#include "Engine/DataTable.h"
#include "ScheduleRow.generated.h"

USTRUCT(BlueprintType)
struct FScheduleRow : public FTableRowBase
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schedule")
    int32 Hour = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schedule")
    int32 Minute = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schedule")
    FName TargetPointName;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Schedule")
    FName ActivityName;
};
