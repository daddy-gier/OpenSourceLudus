#include "NH_ScheduleComponent.h"

#include "Engine/TargetPoint.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"

UNH_ScheduleComponent::UNH_ScheduleComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UNH_ScheduleComponent::RefreshScheduleAtTime(int32 Hour, int32 Minute)
{
    if (!ScheduleDataTable)
    {
        return;
    }

    const TMap<FName, uint8*>& RowMap = ScheduleDataTable->GetRowMap();
    for (const TPair<FName, uint8*>& Pair : RowMap)
    {
        const FScheduleRow* Row = reinterpret_cast<FScheduleRow*>(Pair.Value);
        if (!Row)
        {
            continue;
        }

        if (Row->Hour == Hour && Row->Minute == Minute)
        {
            ApplyScheduleRow(*Row);
            return;
        }
    }
}

void UNH_ScheduleComponent::ForceOverride(FName TargetPointName, FName ActivityName)
{
    FScheduleRow OverrideRow;
    OverrideRow.TargetPointName = TargetPointName;
    OverrideRow.ActivityName = ActivityName;
    ApplyScheduleRow(OverrideRow);
}

void UNH_ScheduleComponent::ApplyScheduleRow(const FScheduleRow& Row)
{
    AActor* TargetActor = FindTargetPointByName(Row.TargetPointName);
    OnScheduleUpdated.Broadcast(Row.ActivityName, TargetActor);
}

AActor* UNH_ScheduleComponent::FindTargetPointByName(FName TargetPointName) const
{
    if (!GetWorld() || TargetPointName.IsNone())
    {
        return nullptr;
    }

    for (TActorIterator<ATargetPoint> It(GetWorld()); It; ++It)
    {
        if (It->GetFName() == TargetPointName)
        {
            return *It;
        }
    }

    return nullptr;
}
