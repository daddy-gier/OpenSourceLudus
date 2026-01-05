#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "NHTypes.h"
#include "NHContractSubsystem.generated.h"

UCLASS()
class NYGHTSHADEHOLLOWCORE_API UNHContractSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Contracts")
    FGuid CreateContract(ENHContractType Type, FName TargetIdTag, int32 PriceDC, float VisibleChance, const FString& Notes);

    UFUNCTION(BlueprintCallable, Category = "Contracts")
    void AssignContract(const FGuid& ContractId, FName ContractorActorTag);

    UFUNCTION(BlueprintCallable, Category = "Contracts")
    void TickContracts(float DeltaSeconds);

    UFUNCTION(BlueprintCallable, Category = "Contracts")
    void ResolveContractNow(const FGuid& ContractId);

    UFUNCTION(BlueprintCallable, Category = "Contracts")
    TArray<FNHContract> GetContracts() const;

private:
    UPROPERTY()
    TArray<FNHContract> Contracts;

    UPROPERTY()
    TMap<FGuid, float> ContractTimers;

    void ApplyProtectedTargetRule(FNHContract& Contract);
    void ResolveContract(FNHContract& Contract);
};
