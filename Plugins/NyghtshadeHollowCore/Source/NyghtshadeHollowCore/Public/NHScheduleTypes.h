#pragma once

#include "CoreMinimal.h"
#include "NHTypes.h"
#include "NHScheduleTypes.generated.h"

USTRUCT(BlueprintType)
struct FNHScheduleTarget
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName TargetTag = NAME_None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TWeakObjectPtr<AActor> CachedActor;
};
