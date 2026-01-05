#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameStateBase.h"
#include "NHTimeGameState.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FNHOnTimeChanged, int32, CurrentDay, int32, CurrentMinuteOfDay);

UCLASS()
class NYGHTSHADEHOLLOWCORE_API ANHTimeGameState : public AGameStateBase
{
    GENERATED_BODY()

public:
    ANHTimeGameState();

    virtual void Tick(float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Time")
    float MinutesPerRealSecond;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Time")
    int32 CurrentMinuteOfDay;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Time")
    int32 CurrentDay;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Time")
    bool bPausedTime;

    UPROPERTY(BlueprintAssignable, Category = "Time")
    FNHOnTimeChanged OnTimeChanged;

    UFUNCTION(BlueprintCallable, Category = "Time")
    FString GetTimeHHMM() const;

    UFUNCTION(BlueprintCallable, Category = "Time")
    void SetTimePaused(bool bPaused);

    UFUNCTION(BlueprintCallable, Category = "Time")
    void SetTimeScale(float MinutesPerSecond);

private:
    float MinuteAccumulator;
    void AdvanceMinutes(int32 MinutesToAdvance);
};
