#include "Components/NH_FactionReputationComponent.h"

int32 UNH_FactionReputationComponent::GetReputation(FName FactionName) const
{
    if (const FNHFactionReputation* Reputation = FindFaction(FactionName))
    {
        return Reputation->Reputation;
    }

    return 0;
}

void UNH_FactionReputationComponent::ModifyReputation(FName FactionName, int32 Delta)
{
    FNHFactionReputation* Reputation = FindFaction(FactionName);
    if (!Reputation)
    {
        return;
    }

    const int32 NewValue = FMath::Clamp(Reputation->Reputation + Delta, Reputation->ReputationFloor, Reputation->ReputationCeiling);
    if (NewValue == Reputation->Reputation)
    {
        return;
    }

    Reputation->Reputation = NewValue;
    OnReputationChanged.Broadcast(FactionName, NewValue);
}

FNHFactionReputation* UNH_FactionReputationComponent::FindFaction(FName FactionName)
{
    return Factions.FindByPredicate([FactionName](const FNHFactionReputation& Candidate)
    {
        return Candidate.FactionName == FactionName;
    });
}

const FNHFactionReputation* UNH_FactionReputationComponent::FindFaction(FName FactionName) const
{
    return Factions.FindByPredicate([FactionName](const FNHFactionReputation& Candidate)
    {
        return Candidate.FactionName == FactionName;
    });
}
