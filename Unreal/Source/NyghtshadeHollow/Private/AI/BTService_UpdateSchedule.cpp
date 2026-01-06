#include "AI/BTService_UpdateSchedule.h"
#include "AI/NH_ScheduleComponent.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "AIController.h"

UBTService_UpdateSchedule::UBTService_UpdateSchedule()
{
    Interval = 5.0f;
    RandomDeviation = 0.0f;
    NodeName = TEXT("Update Schedule");
}

void UBTService_UpdateSchedule::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);

    AAIController* Controller = OwnerComp.GetAIOwner();
    if (!Controller)
    {
        return;
    }

    APawn* Pawn = Controller->GetPawn();
    if (!Pawn)
    {
        return;
    }

    UNH_ScheduleComponent* ScheduleComponent = Pawn->FindComponentByClass<UNH_ScheduleComponent>();
    if (!ScheduleComponent)
    {
        return;
    }

    UBlackboardComponent* Blackboard = OwnerComp.GetBlackboardComponent();
    if (!Blackboard)
    {
        return;
    }

    AActor* TargetActor = ScheduleComponent->GetCurrentTargetActor();
    Blackboard->SetValueAsObject(TEXT("TargetActor"), TargetActor);

    if (TargetActor)
    {
        Blackboard->SetValueAsVector(TEXT("TargetLocation"), TargetActor->GetActorLocation());
    }

    Blackboard->SetValueAsName(TEXT("Activity"), ScheduleComponent->GetCurrentActivity());
    Blackboard->SetValueAsInt(TEXT("ScheduleIndex"), ScheduleComponent->GetCurrentScheduleIndex());
}
