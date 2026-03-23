#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NH_FactionReputationComponent.generated.h"

USTRUCT(BlueprintType)
struct FNHFactionReputation
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    FName FactionName;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    int32 Reputation = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    int32 ReputationFloor = -100;

    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    int32 ReputationCeiling = 100;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FNHFactionReputationChanged, FName, FactionName, int32, NewValue);

UCLASS(ClassGroup=(Nyghtshade), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UNH_FactionReputationComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Reputation")
    TArray<FNHFactionReputation> Factions;

    UPROPERTY(BlueprintAssignable, Category="Reputation")
    FNHFactionReputationChanged OnReputationChanged;

    UFUNCTION(BlueprintCallable, Category="Reputation")
    int32 GetReputation(FName FactionName) const;

    UFUNCTION(BlueprintCallable, Category="Reputation")
    void ModifyReputation(FName FactionName, int32 Delta);

private:
    FNHFactionReputation* FindFaction(FName FactionName);
    const FNHFactionReputation* FindFaction(FName FactionName) const;
};
