#include "AI/ScheduleManager.h"
#include "AI/NH_ScheduleComponent.h"
#include "UObject/UObjectIterator.h"
#include "Time/NH_GameState.h"

AScheduleManager::AScheduleManager()
{
    PrimaryActorTick.bCanEverTick = false;
}

void AScheduleManager::BeginPlay()
{
    Super::BeginPlay();

    if (!HasAuthority())
    {
        return;
    }

    if (const UWorld* World = GetWorld())
    {
        if (ANH_GameState* GameState = World->GetGameState<ANH_GameState>())
        {
            GameState->OnMinuteChanged.AddDynamic(this, &AScheduleManager::HandleMinuteChanged);
        }
    }
}

void AScheduleManager::HandleMinuteChanged(int32 Hour, int32 Minute)
{
    if (!HasAuthority())
    {
        return;
    }

    UWorld* World = GetWorld();
    if (!World)
    {
        return;
    }

    for (TObjectIterator<UNH_ScheduleComponent> It; It; ++It)
    {
        if (It->GetWorld() == World)
        {
            It->RefreshScheduleAtTime(Hour, Minute);
        }
    }
}
