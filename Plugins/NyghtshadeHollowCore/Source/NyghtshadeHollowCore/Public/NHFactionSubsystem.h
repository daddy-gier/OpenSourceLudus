#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "NHTypes.h"
#include "NHFactionSubsystem.generated.h"

UCLASS()
class NYGHTSHADEHOLLOWCORE_API UNHFactionSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Factions")
    int32 GetRep(ENHFactionId Faction) const;

    UFUNCTION(BlueprintCallable, Category = "Factions")
    void AddRep(ENHFactionId Faction, int32 Delta);

    UFUNCTION(BlueprintCallable, Category = "Factions")
    void SetRep(ENHFactionId Faction, int32 Value);

    UFUNCTION(BlueprintCallable, Category = "Factions")
    FString GetRepTier(ENHFactionId Faction) const;

private:
    UPROPERTY()
    TMap<ENHFactionId, int32> Reputation;

    int32 ClampRep(int32 Value) const;
};
