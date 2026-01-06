#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameStateBase.h"
#include "NH_GameState.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnNHMinuteChanged, int32, Hour, int32, Minute);

UCLASS()
class NYGHTSHADEHOLLOW_API ANH_GameState : public AGameStateBase
{
    GENERATED_BODY()

public:
    ANH_GameState();

    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Time")
    int32 CurrentHour = 6;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Time")
    int32 CurrentMinute = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Time")
    float TimeScale = 1.0f;

    UPROPERTY(BlueprintAssignable, Category = "Time")
    FOnNHMinuteChanged OnMinuteChanged;

    UFUNCTION(BlueprintCallable, Category = "Time")
    void SetTime(int32 Hour, int32 Minute);

    UFUNCTION(BlueprintCallable, Category = "Time")
    void GetTime(int32& OutHour, int32& OutMinute) const;

protected:
    float MinuteAccumulator = 0.0f;
};
