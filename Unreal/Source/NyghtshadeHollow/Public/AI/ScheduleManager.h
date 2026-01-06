#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ScheduleManager.generated.h"

UCLASS()
class NYGHTSHADEHOLLOW_API AScheduleManager : public AActor
{
    GENERATED_BODY()

public:
    AScheduleManager();

protected:
    virtual void BeginPlay() override;

private:
    UFUNCTION()
    void HandleMinuteChanged(int32 Hour, int32 Minute);
};
