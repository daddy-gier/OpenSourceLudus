#include "CorruptionComponent.h"
#include "Net/UnrealNetwork.h"

void UCorruptionComponent::AddCorruption(float Amount)
{
    CorruptionLevel = FMath::Clamp(CorruptionLevel + Amount, 0.f, 100.f);
}

bool UCorruptionComponent::IsCompromised() const
{
    return CorruptionLevel >= 40.f;
}

void UCorruptionComponent::RegisterBribe(const FBribeRecord& Record)
{
    BribeHistory.Add(Record);
}

void UCorruptionComponent::RegisterAuditFlag(const FString& Reason, float GameTime)
{
    FAuditFlag Flag;
    Flag.Time = GameTime;
    Flag.Reason = Reason;
    AuditFlags.Add(Flag);
}

void UCorruptionComponent::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& Out) const
{
    Super::GetLifetimeReplicatedProps(Out);
    DOREPLIFETIME(UCorruptionComponent, CorruptionLevel);
}
