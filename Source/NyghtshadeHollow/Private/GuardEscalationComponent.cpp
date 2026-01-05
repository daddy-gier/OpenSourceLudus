#include "GuardEscalationComponent.h"
#include "InmateAuthorityComponent.h"
#include "InmateJudicialComponent.h"
#include "NH_ScheduleComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"

void UGuardEscalationComponent::EvaluateInmate(ACharacter* Inmate)
{
    if (!Inmate)
    {
        return;
    }

    auto* Authority = Inmate->FindComponentByClass<UInmateAuthorityComponent>();
    auto* Judicial = Inmate->FindComponentByClass<UInmateJudicialComponent>();
    auto* Schedule = Inmate->FindComponentByClass<UNH_ScheduleComponent>();

    if (!Authority || !Judicial || !Schedule)
    {
        return;
    }

    if (!Schedule->IsAtScheduledLocation())
    {
        Authority->RegisterViolation(
            EViolationType::Schedule,
            1,
            GetOwner() ? GetOwner()->GetFName() : NAME_None,
            TEXT("Not at scheduled location")
        );

        Escalate(Inmate, Authority->AuthorityLevel);
    }
}

void UGuardEscalationComponent::Escalate(ACharacter* Inmate, int32 Level)
{
    if (Level < AuthorityThreshold)
    {
        IssueWarning(Inmate);
        return;
    }

    if (Level >= 3)
    {
        if (auto* Judicial = Inmate->FindComponentByClass<UInmateJudicialComponent>())
        {
            Judicial->SentenceInmate(
                EPunishmentType::SolitarySegregation,
                3600.f,
                TEXT("Repeated schedule violations")
            );
        }

        if (auto* Movement = Inmate->GetCharacterMovement())
        {
            Movement->MaxWalkSpeed = 120.f;
        }
    }
}

void UGuardEscalationComponent::IssueWarning(ACharacter* Inmate)
{
    if (!Inmate || !GetOwner())
    {
        return;
    }

    UE_LOG(LogTemp, Warning, TEXT("Guard %s issued warning to %s"),
        *GetOwner()->GetName(),
        *Inmate->GetName()
    );
}
