#include "NHContractSubsystem.h"

FGuid UNHContractSubsystem::CreateContract(ENHContractType Type, FName TargetIdTag, int32 PriceDC, float VisibleChance, const FString& Notes)
{
    FNHContract Contract;
    Contract.Id = FGuid::NewGuid();
    Contract.Type = Type;
    Contract.TargetIdTag = TargetIdTag;
    Contract.PriceDC = FMath::Max(0, PriceDC);
    Contract.Status = ENHContractStatus::Requested;
    Contract.SuccessChanceVisible = FMath::Clamp(VisibleChance, 0.0f, 1.0f);
    Contract.SuccessChanceHidden = Contract.SuccessChanceVisible;
    Contract.Notes = Notes;

    ApplyProtectedTargetRule(Contract);
    Contracts.Add(Contract);
    ContractTimers.Add(Contract.Id, 0.0f);
    return Contract.Id;
}

void UNHContractSubsystem::AssignContract(const FGuid& ContractId, FName ContractorActorTag)
{
    for (FNHContract& Contract : Contracts)
    {
        if (Contract.Id == ContractId)
        {
            Contract.Status = ENHContractStatus::Assigned;
            Contract.Notes = FString::Printf(TEXT("%s | Assigned to %s"), *Contract.Notes, *ContractorActorTag.ToString());
            return;
        }
    }
}

void UNHContractSubsystem::TickContracts(float DeltaSeconds)
{
    for (FNHContract& Contract : Contracts)
    {
        if (Contract.Status == ENHContractStatus::Assigned || Contract.Status == ENHContractStatus::InProgress)
        {
            float& Timer = ContractTimers.FindOrAdd(Contract.Id);
            Timer += DeltaSeconds;
            if (Timer > 5.0f)
            {
                Contract.Status = ENHContractStatus::InProgress;
            }
            if (Timer > 15.0f)
            {
                ResolveContract(Contract);
                Timer = 0.0f;
            }
        }
    }
}

void UNHContractSubsystem::ResolveContractNow(const FGuid& ContractId)
{
    for (FNHContract& Contract : Contracts)
    {
        if (Contract.Id == ContractId)
        {
            ResolveContract(Contract);
            return;
        }
    }
}

TArray<FNHContract> UNHContractSubsystem::GetContracts() const
{
    return Contracts;
}

void UNHContractSubsystem::ApplyProtectedTargetRule(FNHContract& Contract)
{
    const FString TagString = Contract.TargetIdTag.ToString();
    if (TagString.Equals(TEXT("ARI"), ESearchCase::IgnoreCase)
        || TagString.Equals(TEXT("MARI"), ESearchCase::IgnoreCase)
        || TagString.Equals(TEXT("LEE"), ESearchCase::IgnoreCase))
    {
        Contract.SuccessChanceHidden = 0.0f;
    }
}

void UNHContractSubsystem::ResolveContract(FNHContract& Contract)
{
    ApplyProtectedTargetRule(Contract);
    const float Roll = FMath::FRand();
    if (Roll <= Contract.SuccessChanceHidden)
    {
        Contract.Status = ENHContractStatus::Completed;
    }
    else
    {
        Contract.Status = ENHContractStatus::Failed;
    }
}
