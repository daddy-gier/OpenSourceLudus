#include "NHScheduleComponent.h"
#include "EngineUtils.h"
#include "Kismet/GameplayStatics.h"

UNHScheduleComponent::UNHScheduleComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    CurrentActivityRowName = NAME_None;
    CurrentActivity = FNHActivityRow();
}

void UNHScheduleComponent::BeginPlay()
{
    Super::BeginPlay();

    if (UWorld* World = GetWorld())
    {
        if (ANHTimeGameState* TimeState = World->GetGameState<ANHTimeGameState>())
        {
            TimeState->OnTimeChanged.AddDynamic(this, &UNHScheduleComponent::HandleTimeChanged);
            RecomputeActivity(TimeState->CurrentMinuteOfDay);
        }
    }
}

void UNHScheduleComponent::HandleTimeChanged(int32 CurrentDay, int32 CurrentMinuteOfDay)
{
    RecomputeActivity(CurrentMinuteOfDay);
}

void UNHScheduleComponent::RecomputeActivity(int32 MinuteOfDay)
{
    if (!ScheduleTable)
    {
        return;
    }

    FName FoundRowName = NAME_None;
    FNHActivityRow FoundRow;
    bool bFound = false;

    TArray<FName> RowNames = ScheduleTable->GetRowNames();
    for (const FName& RowName : RowNames)
    {
        const FNHActivityRow* Row = ScheduleTable->FindRow<FNHActivityRow>(RowName, TEXT("ScheduleLookup"));
        if (!Row)
        {
            continue;
        }
        if (MinuteOfDay >= Row->StartMinute && MinuteOfDay < Row->EndMinute)
        {
            FoundRowName = RowName;
            FoundRow = *Row;
            bFound = true;
            break;
        }
    }

    if (!bFound)
    {
        return;
    }

    const bool bChanged = FoundRowName != CurrentActivityRowName;
    CurrentActivityRowName = FoundRowName;
    CurrentActivity = FoundRow;

    if (bChanged)
    {
        AActor* TargetActor = ResolveTargetActor(CurrentActivity.TargetTag);
        OnActivityChanged.Broadcast(CurrentActivity.Activity, TargetActor, CurrentActivityRowName);
    }
}

AActor* UNHScheduleComponent::ResolveTargetActor(const FName& TargetTag) const
{
    if (TargetTag.IsNone())
    {
        return nullptr;
    }

    if (UWorld* World = GetWorld())
    {
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            if (It->ActorHasTag(TargetTag))
            {
                return *It;
            }
        }
    }

    return nullptr;
}

ENHActivityType UNHScheduleComponent::GetCurrentActivityType() const
{
    return CurrentActivity.Activity;
}

AActor* UNHScheduleComponent::GetCurrentTargetActor() const
{
    return ResolveTargetActor(CurrentActivity.TargetTag);
}

void UNHScheduleComponent::ForceRecomputeNow()
{
    if (UWorld* World = GetWorld())
    {
        if (ANHTimeGameState* TimeState = World->GetGameState<ANHTimeGameState>())
        {
            RecomputeActivity(TimeState->CurrentMinuteOfDay);
        }
    }
}
