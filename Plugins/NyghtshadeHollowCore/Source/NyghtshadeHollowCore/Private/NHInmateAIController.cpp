#include "NHInmateAIController.h"

void ANHInmateAIController::OnPossess(APawn* InPawn)
{
    Super::OnPossess(InPawn);

    CurrentMoveTarget = nullptr;

    if (!InPawn)
    {
        return;
    }

    if (UNHScheduleComponent* Schedule = InPawn->FindComponentByClass<UNHScheduleComponent>())
    {
        Schedule->OnActivityChanged.AddDynamic(this, &ANHInmateAIController::HandleActivityChanged);
    }
}

void ANHInmateAIController::HandleActivityChanged(ENHActivityType ActivityType, AActor* TargetActor, FName RowName)
{
    CurrentMoveTarget = TargetActor;
}

AActor* ANHInmateAIController::GetCurrentMoveTarget() const
{
    return CurrentMoveTarget;
}
