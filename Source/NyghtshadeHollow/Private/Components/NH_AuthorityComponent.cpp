#include "Components/NH_AuthorityComponent.h"

UNH_AuthorityComponent::UNH_AuthorityComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UNH_AuthorityComponent::BeginPlay()
{
    Super::BeginPlay();
}

void UNH_AuthorityComponent::AddViolation(const FNHViolationRecord& Violation, bool bAutoEscalate)
{
    ViolationHistory.Add(Violation);

    if (bAutoEscalate)
    {
        const ENHAuthorityLevel PreviousLevel = CurrentAuthorityLevel;
        const int32 NextLevelIndex = FMath::Min(static_cast<int32>(CurrentAuthorityLevel) + 1, static_cast<int32>(ENHAuthorityLevel::InstitutionalRisk));
        SetAuthorityLevel(static_cast<ENHAuthorityLevel>(NextLevelIndex));

        if (PreviousLevel != CurrentAuthorityLevel)
        {
            OnAuthorityLevelChanged.Broadcast(PreviousLevel, CurrentAuthorityLevel);
        }
    }
}

void UNH_AuthorityComponent::SetAuthorityLevel(ENHAuthorityLevel NewLevel)
{
    if (CurrentAuthorityLevel == NewLevel)
    {
        return;
    }

    const ENHAuthorityLevel PreviousLevel = CurrentAuthorityLevel;
    CurrentAuthorityLevel = NewLevel;
    OnAuthorityLevelChanged.Broadcast(PreviousLevel, CurrentAuthorityLevel);
}

void UNH_AuthorityComponent::BeginPunishment(const FNHPunishmentRecord& NewPunishment)
{
    ActivePunishments.Add(NewPunishment);
    OnPunishmentStarted.Broadcast(NewPunishment);
}

void UNH_AuthorityComponent::EndPunishmentByType(FName PunishmentType)
{
    for (int32 Index = ActivePunishments.Num() - 1; Index >= 0; --Index)
    {
        if (ActivePunishments[Index].PunishmentType == PunishmentType)
        {
            FNHPunishmentRecord EndedPunishment = ActivePunishments[Index];
            EndedPunishment.bIsActive = false;
            ActivePunishments.RemoveAt(Index);
            OnPunishmentEnded.Broadcast(EndedPunishment);
        }
    }
}

bool UNH_AuthorityComponent::HasActivePunishment(FName PunishmentType) const
{
    return ActivePunishments.ContainsByPredicate([PunishmentType](const FNHPunishmentRecord& Record)
    {
        return Record.PunishmentType == PunishmentType && Record.bIsActive;
    });
}

FNHPunishmentRecord UNH_AuthorityComponent::GetActivePunishment(FName PunishmentType, bool& bFound) const
{
    const FNHPunishmentRecord* Record = ActivePunishments.FindByPredicate([PunishmentType](const FNHPunishmentRecord& Candidate)
    {
        return Candidate.PunishmentType == PunishmentType && Candidate.bIsActive;
    });

    if (Record)
    {
        bFound = true;
        return *Record;
    }

    bFound = false;
    return FNHPunishmentRecord();
}

void UNH_AuthorityComponent::TickPunishments(float DeltaSeconds)
{
    for (int32 Index = ActivePunishments.Num() - 1; Index >= 0; --Index)
    {
        FNHPunishmentRecord& Record = ActivePunishments[Index];
        if (!Record.bIsActive)
        {
            continue;
        }

        Record.TimeServed += DeltaSeconds;

        if (Record.Duration > 0.0f && Record.TimeServed >= Record.Duration)
        {
            Record.bIsActive = false;
            FNHPunishmentRecord EndedPunishment = Record;
            ActivePunishments.RemoveAt(Index);
            OnPunishmentEnded.Broadcast(EndedPunishment);
        }
    }
}
