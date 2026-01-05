#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/DataTable.h"
#include "NHTypes.h"
#include "NHTimeGameState.h"
#include "NHScheduleComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(
    FNHScheduleChanged,
    ENHActivityType,
    ActivityType,
    AActor*,
    TargetActor,
    FName,
    RowName
);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOWCORE_API UNHScheduleComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UNHScheduleComponent();

    virtual void BeginPlay() override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schedule")
    UDataTable* ScheduleTable;

    UPROPERTY(BlueprintAssignable, Category = "Schedule")
    FNHScheduleChanged OnActivityChanged;

    UFUNCTION(BlueprintCallable, Category = "Schedule")
    ENHActivityType GetCurrentActivityType() const;

    UFUNCTION(BlueprintCallable, Category = "Schedule")
    AActor* GetCurrentTargetActor() const;

    UFUNCTION(BlueprintCallable, Category = "Schedule")
    void ForceRecomputeNow();

protected:
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schedule")
    FName CurrentActivityRowName;

    UPROPERTY()
    FNHActivityRow CurrentActivity;

private:
    UFUNCTION()
    void HandleTimeChanged(int32 CurrentDay, int32 CurrentMinuteOfDay);

    void RecomputeActivity(int32 MinuteOfDay);
    AActor* ResolveTargetActor(const FName& TargetTag) const;
};
