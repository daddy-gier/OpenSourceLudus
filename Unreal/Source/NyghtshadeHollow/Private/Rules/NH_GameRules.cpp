#include "Rules/NH_GameRules.h"

ANH_GameRules::ANH_GameRules()
{
    PrimaryActorTick.bCanEverTick = false;
}

void ANH_GameRules::NotifyCrime(AActor* Perpetrator, int32 Severity)
{
    OnCrimeReported.Broadcast(Perpetrator, Severity);
}

void ANH_GameRules::StartLockdown()
{
    if (!bLockdownActive)
    {
        bLockdownActive = true;
        OnLockdownChanged.Broadcast(true);
    }
}

void ANH_GameRules::EndLockdown()
{
    if (bLockdownActive)
    {
        bLockdownActive = false;
        OnLockdownChanged.Broadcast(false);
    }
}

bool ANH_GameRules::IsLockdownActive() const
{
    return bLockdownActive;
}
