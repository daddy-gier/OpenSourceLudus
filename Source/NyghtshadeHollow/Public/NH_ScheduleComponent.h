#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/DataTable.h"
#include "ScheduleRow.h"
#include "NH_ScheduleComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnScheduleUpdated, FName, ActivityName, AActor*, TargetActor);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UNH_ScheduleComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UNH_ScheduleComponent();

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Schedule")
    FName ScheduleId;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Schedule")
    UDataTable* ScheduleDataTable = nullptr;

    UPROPERTY(BlueprintAssignable, Category="Schedule")
    FOnScheduleUpdated OnScheduleUpdated;

    UFUNCTION(BlueprintCallable, Category="Schedule")
    void RefreshScheduleAtTime(int32 Hour, int32 Minute);

    UFUNCTION(BlueprintCallable, Category="Schedule")
    void ForceOverride(FName TargetPointName, FName ActivityName);

private:
    void ApplyScheduleRow(const FScheduleRow& Row);
    AActor* FindTargetPointByName(FName TargetPointName) const;
};
