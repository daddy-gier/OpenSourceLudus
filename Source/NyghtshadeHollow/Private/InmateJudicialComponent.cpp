#include "InmateJudicialComponent.h"
#include "Net/UnrealNetwork.h"

UInmateJudicialComponent::UInmateJudicialComponent()
{
    SetIsReplicatedByDefault(true);
}

void UInmateJudicialComponent::SentenceInmate(
    EPunishmentType Type,
    float Duration,
    const FString& Reason)
{
    FPunishmentRecord Record;
    Record.Type = Type;
    Record.Duration = Duration;
    Record.TimeServed = 0.f;
    Record.StartGameTime = GetWorld() ? GetWorld()->GetTimeSeconds() : 0.f;
    Record.bEligibleForReview = (Duration > 0);
    Record.Reason = Reason;

    ActivePunishments.Add(Record);
    OnPunishmentAdded.Broadcast(Record);
}

void UInmateJudicialComponent::TickPunishments(float DeltaSeconds)
{
    for (int32 i = ActivePunishments.Num() - 1; i >= 0; --i)
    {
        auto& P = ActivePunishments[i];
        if (P.Duration < 0.f)
        {
            continue;
        }

        P.TimeServed += DeltaSeconds;
        if (P.TimeServed >= P.Duration)
        {
            ActivePunishments.RemoveAt(i);
        }
    }
}

bool UInmateJudicialComponent::IsInSegregation() const
{
    for (const auto& P : ActivePunishments)
    {
        if (P.Type == EPunishmentType::SolitarySegregation ||
            P.Type == EPunishmentType::IndefiniteSegregation)
        {
            return true;
        }
    }
    return false;
}

void UInmateJudicialComponent::GetLifetimeReplicatedProps(
    TArray<FLifetimeProperty>& Out) const
{
    Super::GetLifetimeReplicatedProps(Out);
    DOREPLIFETIME(UInmateJudicialComponent, ActivePunishments);
}
