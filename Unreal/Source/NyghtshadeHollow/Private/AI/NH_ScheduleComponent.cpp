#include "AI/NH_ScheduleComponent.h"
#include "Engine/TargetPoint.h"
#include "EngineUtils.h"
#include "Kismet/GameplayStatics.h"
#include "Time/NH_GameState.h"

UNH_ScheduleComponent::UNH_ScheduleComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UNH_ScheduleComponent::BeginPlay()
{
    Super::BeginPlay();

    if (const UWorld* World = GetWorld())
    {
        if (ANH_GameState* GameState = World->GetGameState<ANH_GameState>())
        {
            GameState->OnMinuteChanged.AddDynamic(this, &UNH_ScheduleComponent::HandleMinuteChanged);
        }
    }
}

void UNH_ScheduleComponent::RefreshScheduleAtTime(int32 Hour, int32 Minute)
{
    if (!ScheduleDataTable)
    {
        return;
    }

    TArray<FScheduleRow*> Rows;
    ScheduleDataTable->GetAllRows(TEXT("ScheduleLookup"), Rows);

    int32 RowIndex = 0;
    for (const FScheduleRow* Row : Rows)
    {
        if (!Row || Row->NPCId != ScheduleId)
        {
            RowIndex++;
            continue;
        }

        if (IsRowActive(*Row, Hour, Minute))
        {
            if (CurrentScheduleIndex != RowIndex || CurrentActivity != Row->ActivityName)
            {
                CurrentScheduleIndex = RowIndex;
                CurrentActivity = Row->ActivityName;
                CurrentTargetActor = FindTargetActor(Row->TargetPointName);
                OnScheduleUpdated.Broadcast(CurrentActivity, CurrentTargetActor.Get());
            }
            return;
        }

        RowIndex++;
    }
}

FName UNH_ScheduleComponent::GetCurrentActivity() const
{
    return CurrentActivity;
}

AActor* UNH_ScheduleComponent::GetCurrentTargetActor() const
{
    return CurrentTargetActor.Get();
}

int32 UNH_ScheduleComponent::GetCurrentScheduleIndex() const
{
    return CurrentScheduleIndex;
}

void UNH_ScheduleComponent::HandleMinuteChanged(int32 Hour, int32 Minute)
{
    RefreshScheduleAtTime(Hour, Minute);
}

bool UNH_ScheduleComponent::IsRowActive(const FScheduleRow& Row, int32 Hour, int32 Minute) const
{
    const int32 Current = (Hour * 60) + Minute;
    const int32 Start = (Row.StartHour * 60) + Row.StartMinute;
    const int32 End = (Row.EndHour * 60) + Row.EndMinute;

    if (Start <= End)
    {
        return Current >= Start && Current < End;
    }

    return Current >= Start || Current < End;
}

AActor* UNH_ScheduleComponent::FindTargetActor(const FName& TargetName) const
{
    if (TargetName.IsNone())
    {
        return nullptr;
    }

    UWorld* World = GetWorld();
    if (!World)
    {
        return nullptr;
    }

    for (TActorIterator<ATargetPoint> It(World); It; ++It)
    {
        if (It->GetFName() == TargetName)
        {
            return *It;
        }
    }

    return nullptr;
}
