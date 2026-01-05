#include "InmateAuthorityComponent.h"
#include "Net/UnrealNetwork.h"

UInmateAuthorityComponent::UInmateAuthorityComponent()
{
    SetIsReplicatedByDefault(true);
    AuthorityLevel = 0;
}

void UInmateAuthorityComponent::RegisterViolation(
    EViolationType Type,
    int32 Severity,
    FName ReportingGuard,
    const FString& Context)
{
    FViolationRecord Record;
    Record.Type = Type;
    Record.Severity = Severity;
    Record.ReportingGuard = ReportingGuard;
    Record.Context = Context;
    Record.GameTimeStamp = GetWorld() ? GetWorld()->GetTimeSeconds() : 0.f;

    ViolationHistory.Add(Record);

    AuthorityLevel = FMath::Clamp(AuthorityLevel + Severity, 0, 5);
}

void UInmateAuthorityComponent::DecayAuthority(float DeltaGameTime)
{
    if (AuthorityLevel > 0 && DeltaGameTime > 600.f)
    {
        AuthorityLevel--;
    }
}

void UInmateAuthorityComponent::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& Out) const
{
    Super::GetLifetimeReplicatedProps(Out);
    DOREPLIFETIME(UInmateAuthorityComponent, AuthorityLevel);
    DOREPLIFETIME(UInmateAuthorityComponent, ViolationHistory);
}
