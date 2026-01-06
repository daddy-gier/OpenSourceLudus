#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTTaskNode.h"
#include "BTTask_PerformActivity.generated.h"

UCLASS()
class NYGHTSHADEHOLLOW_API UBTTask_PerformActivity : public UBTTaskNode
{
    GENERATED_BODY()

public:
    UBTTask_PerformActivity();

    UPROPERTY(EditAnywhere, Category = "Activity")
    float ActivityDuration = 2.0f;

protected:
    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;
    virtual void TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

private:
    float RemainingTime = 0.0f;
    FName CachedActivityName;
};
