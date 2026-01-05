#include "ScheduleManager.h"

#include "EngineUtils.h"
#include "NH_GameState.h"
#include "NH_ScheduleComponent.h"

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

    if (ANH_GameState* GS = GetWorld()->GetGameState<ANH_GameState>())
    {
        GS->OnMinuteChanged.AddDynamic(this, &AScheduleManager::OnMinuteChanged);
    }
}

void AScheduleManager::OnMinuteChanged(int32 Hour, int32 Minute)
{
    RefreshAllSchedules(Hour, Minute);
}

void AScheduleManager::RefreshAllSchedules(int32 Hour, int32 Minute)
{
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        if (UNH_ScheduleComponent* Schedule = It->FindComponentByClass<UNH_ScheduleComponent>())
        {
            if (bLockdownActive)
            {
                Schedule->ForceOverride(LockdownTargetPoint, "Lockdown");
            }
            else
            {
                Schedule->RefreshScheduleAtTime(Hour, Minute);
            }
        }
    }
}

void AScheduleManager::ForceTime(int32 Hour, int32 Minute)
{
    if (ANH_GameState* GS = GetWorld()->GetGameState<ANH_GameState>())
    {
        GS->SetTime(Hour, Minute);
    }
}

void AScheduleManager::StartLockdown()
{
    bLockdownActive = true;

    int32 Hour = 0;
    int32 Minute = 0;
    if (GetCurrentTime(Hour, Minute))
    {
        RefreshAllSchedules(Hour, Minute);
    }
}

void AScheduleManager::EndLockdown()
{
    bLockdownActive = false;

    int32 Hour = 0;
    int32 Minute = 0;
    if (GetCurrentTime(Hour, Minute))
    {
        RefreshAllSchedules(Hour, Minute);
    }
}

bool AScheduleManager::GetCurrentTime(int32& OutHour, int32& OutMinute) const
{
    if (const ANH_GameState* GS = GetWorld()->GetGameState<ANH_GameState>())
    {
        GS->GetTime(OutHour, OutMinute);
        return true;
    }

    return false;
}
