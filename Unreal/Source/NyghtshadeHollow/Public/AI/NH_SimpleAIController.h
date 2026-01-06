#pragma once

#include "CoreMinimal.h"
#include "AIController.h"
#include "BehaviorTree/BehaviorTree.h"
#include "NH_SimpleAIController.generated.h"

UCLASS()
class NYGHTSHADEHOLLOW_API ANH_SimpleAIController : public AAIController
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "AI")
    UBehaviorTree* BehaviorTreeAsset;

protected:
    virtual void OnPossess(APawn* InPawn) override;
};
