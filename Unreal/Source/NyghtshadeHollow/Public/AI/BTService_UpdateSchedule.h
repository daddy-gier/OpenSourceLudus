#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTService.h"
#include "BTService_UpdateSchedule.generated.h"

UCLASS()
class NYGHTSHADEHOLLOW_API UBTService_UpdateSchedule : public UBTService
{
    GENERATED_BODY()

public:
    UBTService_UpdateSchedule();

protected:
    virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
};
