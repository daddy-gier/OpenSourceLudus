#include "InspectorAuthoritySubsystem.h"

void UInspectorAuthoritySubsystem::AddFlagRecord(EInspectorFlag Flag, float Weight, const FString& Source, float GameTime)
{
    FInspectorFlagRecord Record;
    Record.Flag = Flag;
    Record.Weight = Weight;
    Record.Timestamp = GameTime;
    Record.Source = Source;
    FlagHistory.Add(Record);

    EvaluateFlags(GameTime);
}

float UInspectorAuthoritySubsystem::GetAggregatedScore(EInspectorFlag Flag) const
{
    const float* Score = AggregatedScores.Find(Flag);
    return Score ? *Score : 0.f;
}

const TArray<FInspectorFlagRecord>& UInspectorAuthoritySubsystem::GetFlagHistory() const
{
    return FlagHistory;
}

const TArray<FAuthorizedEventRecord>& UInspectorAuthoritySubsystem::GetAuthorizedEvents() const
{
    return AuthorizedEvents;
}

void UInspectorAuthoritySubsystem::EvaluateFlags(float GameTime)
{
    DecayFlags(GameTime);
    AggregateFlags(GameTime);
    AuthorizeEvents(GameTime);
}

bool UInspectorAuthoritySubsystem::IsEventAuthorized(EAuthorizedNarrativeEvent Event, float GameTime) const
{
    for (const FAuthorizedEventRecord& Record : AuthorizedEvents)
    {
        if (Record.Event == Event && Record.CooldownEndTime > GameTime)
        {
            return true;
        }
    }

    return false;
}

void UInspectorAuthoritySubsystem::RegisterAuthorizedEvent(EAuthorizedNarrativeEvent Event, float GameTime, float CooldownSeconds)
{
    if (IsEventAuthorized(Event, GameTime))
    {
        return;
    }

    FAuthorizedEventRecord Record;
    Record.Event = Event;
    Record.AuthorizedTime = GameTime;
    Record.CooldownEndTime = GameTime + CooldownSeconds;
    AuthorizedEvents.Add(Record);
}

void UInspectorAuthoritySubsystem::DecayFlags(float GameTime)
{
    if (FlagDecayHalfLifeSeconds <= 0.f)
    {
        return;
    }

    const float CutoffTime = GameTime - FlagWindowSeconds;
    FlagHistory.RemoveAll([CutoffTime](const FInspectorFlagRecord& Record)
    {
        return Record.Timestamp < CutoffTime;
    });
}

void UInspectorAuthoritySubsystem::AggregateFlags(float GameTime)
{
    AggregatedScores.Empty();

    const float CutoffTime = GameTime - FlagWindowSeconds;
    for (const FInspectorFlagRecord& Record : FlagHistory)
    {
        if (Record.Timestamp < CutoffTime)
        {
            continue;
        }

        const float AgeSeconds = FMath::Max(GameTime - Record.Timestamp, 0.f);
        const float DecayFactor = FMath::Exp2(-AgeSeconds / FlagDecayHalfLifeSeconds);
        const float WeightedScore = Record.Weight * DecayFactor;

        float& Score = AggregatedScores.FindOrAdd(Record.Flag);
        Score += WeightedScore;
    }

    const float ViolenceScore = GetAggregatedScore(EInspectorFlag::ViolenceSpike);
    const float CorruptionScore = GetAggregatedScore(EInspectorFlag::CorruptionTrend);
    if (ViolenceScore > 0.f && CorruptionScore > 0.f)
    {
        float& AuditRiskScore = AggregatedScores.FindOrAdd(EInspectorFlag::AuditRisk);
        AuditRiskScore += (ViolenceScore * 0.25f) + (CorruptionScore * 0.25f);
    }
}

void UInspectorAuthoritySubsystem::AuthorizeEvents(float GameTime)
{
    const bool bCellSweepAuthorized =
        GetAggregatedScore(EInspectorFlag::ViolenceSpike) >= 3.f &&
        GetAggregatedScore(EInspectorFlag::CorruptionTrend) >= 1.5f;

    if (bCellSweepAuthorized)
    {
        RegisterAuthorizedEvent(EAuthorizedNarrativeEvent::CellSweep, GameTime, 6.f * 3600.f);
    }

    const bool bExternalAuditAuthorized =
        GetAggregatedScore(EInspectorFlag::AuditRisk) >= 4.f &&
        GetAggregatedScore(EInspectorFlag::EvidenceTampering) >= 1.f;

    if (bExternalAuditAuthorized)
    {
        RegisterAuthorizedEvent(EAuthorizedNarrativeEvent::ExternalAuditPing, GameTime, 12.f * 3600.f);
    }
}
