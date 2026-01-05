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

    virtual void BeginPlay() override;

    UPROPERTY(BlueprintReadOnly, Category="Schedule")
    bool bLockdownActive = false;

    UPROPERTY(EditAnywhere, Category="Schedule")
    FName LockdownTargetPoint = "TP_Cell";

    UFUNCTION(Exec)
    void ForceTime(int32 Hour, int32 Minute);

    UFUNCTION(Exec)
    void StartLockdown();

    UFUNCTION(Exec)
    void EndLockdown();

private:
    UFUNCTION()
    void OnMinuteChanged(int32 Hour, int32 Minute);

    void RefreshAllSchedules(int32 Hour, int32 Minute);
    bool GetCurrentTime(int32& OutHour, int32& OutMinute) const;
};
