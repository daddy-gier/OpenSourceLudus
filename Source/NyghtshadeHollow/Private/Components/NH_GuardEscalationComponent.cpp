#include "Components/NH_GuardEscalationComponent.h"
#include "Components/NH_AuthorityComponent.h"
#include "GameFramework/Character.h"

UNH_GuardEscalationComponent::UNH_GuardEscalationComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UNH_GuardEscalationComponent::BeginPlay()
{
    Super::BeginPlay();
}

void UNH_GuardEscalationComponent::EvaluateInmate(ACharacter* Inmate, ENHViolationType ViolationType, int32 Severity, const FString& Context)
{
    if (!Inmate)
    {
        return;
    }

    UNH_AuthorityComponent* AuthorityComponent = Inmate->FindComponentByClass<UNH_AuthorityComponent>();
    if (!AuthorityComponent)
    {
        return;
    }

    FNHViolationRecord Record;
    Record.Type = ViolationType;
    Record.Severity = Severity;
    Record.GameTimeStamp = GetWorld() ? GetWorld()->GetTimeSeconds() : 0.0f;
    Record.ReportingGuard = GetOwner() ? GetOwner()->GetFName() : NAME_None;
    Record.Context = Context;

    AuthorityComponent->AddViolation(Record, true);
    OnViolationLogged.Broadcast(Inmate, Record);

    ENHGuardResponse Response = DetermineResponse(AuthorityComponent, Severity);
    ApplyResponse(Inmate, AuthorityComponent, Response);
}

ENHGuardResponse UNH_GuardEscalationComponent::DetermineResponse(const UNH_AuthorityComponent* AuthorityComponent, int32 Severity) const
{
    if (!AuthorityComponent)
    {
        return ENHGuardResponse::None;
    }

    const int32 AuthorityLevel = static_cast<int32>(AuthorityComponent->CurrentAuthorityLevel);
    const int32 WeightedSeverity = Severity + AuthorityLevel;

    if (WeightedSeverity >= AuthorityThreshold + 3)
    {
        return ENHGuardResponse::Segregation;
    }

    if (WeightedSeverity >= AuthorityThreshold + 2)
    {
        return ENHGuardResponse::Restraint;
    }

    if (WeightedSeverity >= AuthorityThreshold + 1)
    {
        return ENHGuardResponse::Supervision;
    }

    return ENHGuardResponse::VerbalWarning;
}

void UNH_GuardEscalationComponent::ApplyResponse(ACharacter* Inmate, UNH_AuthorityComponent* AuthorityComponent, ENHGuardResponse Response)
{
    if (!Inmate || !AuthorityComponent)
    {
        return;
    }

    CurrentTarget = Inmate;

    switch (Response)
    {
        case ENHGuardResponse::VerbalWarning:
            AuthorityComponent->SetAuthorityLevel(ENHAuthorityLevel::VerballyWarned);
            break;
        case ENHGuardResponse::Supervision:
            AuthorityComponent->SetAuthorityLevel(ENHAuthorityLevel::UnderSupervision);
            break;
        case ENHGuardResponse::Restraint:
            AuthorityComponent->SetAuthorityLevel(ENHAuthorityLevel::Restrained);
            break;
        case ENHGuardResponse::Segregation:
            AuthorityComponent->SetAuthorityLevel(ENHAuthorityLevel::Segregated);
            break;
        default:
            break;
    }

    OnGuardResponseIssued.Broadcast(Inmate, Response);
}
