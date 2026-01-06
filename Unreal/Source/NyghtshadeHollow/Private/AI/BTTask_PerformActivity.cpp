#include "AI/BTTask_PerformActivity.h"
#include "AI/ActivityInterface.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"

UBTTask_PerformActivity::UBTTask_PerformActivity()
{
    bNotifyTick = true;
    NodeName = TEXT("Perform Activity");
}

EBTNodeResult::Type UBTTask_PerformActivity::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    RemainingTime = ActivityDuration;
    CachedActivityName = NAME_None;

    if (const UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent())
    {
        CachedActivityName = Blackboard->GetValueAsName(TEXT("Activity"));
    }

    if (AAIController* Controller = OwnerComp.GetAIOwner())
    {
        if (APawn* Pawn = Controller->GetPawn())
        {
            if (Pawn->GetClass()->ImplementsInterface(UActivityInterface::StaticClass()))
            {
                IActivityInterface::Execute_OnActivityStarted(Pawn, CachedActivityName);
            }
        }
    }

    return EBTNodeResult::InProgress;
}

void UBTTask_PerformActivity::TickTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    RemainingTime -= DeltaSeconds;

    if (RemainingTime > 0.0f)
    {
        return;
    }

    if (AAIController* Controller = OwnerComp.GetAIOwner())
    {
        if (APawn* Pawn = Controller->GetPawn())
        {
            if (Pawn->GetClass()->ImplementsInterface(UActivityInterface::StaticClass()))
            {
                IActivityInterface::Execute_OnActivityEnded(Pawn, CachedActivityName);
            }
        }
    }

    FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
}
