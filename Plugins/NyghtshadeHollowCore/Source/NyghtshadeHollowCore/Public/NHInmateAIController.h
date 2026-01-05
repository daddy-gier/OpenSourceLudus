#pragma once

#include "CoreMinimal.h"
#include "AIController.h"
#include "NHScheduleComponent.h"
#include "NHInmateAIController.generated.h"

UCLASS()
class NYGHTSHADEHOLLOWCORE_API ANHInmateAIController : public AAIController
{
    GENERATED_BODY()

public:
    virtual void OnPossess(APawn* InPawn) override;

    UFUNCTION(BlueprintCallable, Category = "AI")
    AActor* GetCurrentMoveTarget() const;

private:
    UPROPERTY()
    AActor* CurrentMoveTarget;

    UFUNCTION()
    void HandleActivityChanged(ENHActivityType ActivityType, AActor* TargetActor, FName RowName);
};
