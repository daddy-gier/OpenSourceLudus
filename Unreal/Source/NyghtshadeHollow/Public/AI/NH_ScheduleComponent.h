#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/DataTable.h"
#include "AI/ScheduleRow.h"
#include "NH_ScheduleComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnNHUpdatedSchedule, FName, ActivityName, AActor*, TargetActor);

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UNH_ScheduleComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UNH_ScheduleComponent();

    virtual void BeginPlay() override;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Schedule")
    FName ScheduleId;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Schedule")
    UDataTable* ScheduleDataTable;

    UPROPERTY(BlueprintAssignable, Category = "Schedule")
    FOnNHUpdatedSchedule OnScheduleUpdated;

    UFUNCTION(BlueprintCallable, Category = "Schedule")
    void RefreshScheduleAtTime(int32 Hour, int32 Minute);

    UFUNCTION(BlueprintCallable, Category = "Schedule")
    FName GetCurrentActivity() const;

    UFUNCTION(BlueprintCallable, Category = "Schedule")
    AActor* GetCurrentTargetActor() const;

    UFUNCTION(BlueprintCallable, Category = "Schedule")
    int32 GetCurrentScheduleIndex() const;

private:
    void HandleMinuteChanged(int32 Hour, int32 Minute);
    bool IsRowActive(const FScheduleRow& Row, int32 Hour, int32 Minute) const;
    AActor* FindTargetActor(const FName& TargetName) const;

    FName CurrentActivity;
    TWeakObjectPtr<AActor> CurrentTargetActor;
    int32 CurrentScheduleIndex = INDEX_NONE;
};
