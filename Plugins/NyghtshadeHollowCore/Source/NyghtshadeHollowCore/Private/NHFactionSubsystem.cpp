#include "NHFactionSubsystem.h"

int32 UNHFactionSubsystem::ClampRep(int32 Value) const
{
    return FMath::Clamp(Value, -100, 100);
}

int32 UNHFactionSubsystem::GetRep(ENHFactionId Faction) const
{
    if (const int32* Found = Reputation.Find(Faction))
    {
        return *Found;
    }
    return 0;
}

void UNHFactionSubsystem::AddRep(ENHFactionId Faction, int32 Delta)
{
    const int32 Current = GetRep(Faction);
    Reputation.Add(Faction, ClampRep(Current + Delta));
}

void UNHFactionSubsystem::SetRep(ENHFactionId Faction, int32 Value)
{
    Reputation.Add(Faction, ClampRep(Value));
}

FString UNHFactionSubsystem::GetRepTier(ENHFactionId Faction) const
{
    const int32 Value = GetRep(Faction);
    if (Value <= -40)
    {
        return TEXT("Hostile");
    }
    if (Value >= 40)
    {
        return TEXT("Friendly");
    }
    return TEXT("Neutral");
}
