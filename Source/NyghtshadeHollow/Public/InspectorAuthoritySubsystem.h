#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "InspectorTypes.h"
#include "InspectorAuthoritySubsystem.generated.h"

UCLASS()
class NYGHTSHADEHOLLOW_API UInspectorAuthoritySubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable)
    void AddFlagRecord(EInspectorFlag Flag, float Weight, const FString& Source, float GameTime);

    UFUNCTION(BlueprintCallable)
    float GetAggregatedScore(EInspectorFlag Flag) const;

    UFUNCTION(BlueprintCallable)
    const TArray<FInspectorFlagRecord>& GetFlagHistory() const;

    UFUNCTION(BlueprintCallable)
    const TArray<FAuthorizedEventRecord>& GetAuthorizedEvents() const;

    UFUNCTION(BlueprintCallable)
    void EvaluateFlags(float GameTime);

    UFUNCTION(BlueprintCallable)
    bool IsEventAuthorized(EAuthorizedNarrativeEvent Event, float GameTime) const;

    UFUNCTION(BlueprintCallable)
    void RegisterAuthorizedEvent(EAuthorizedNarrativeEvent Event, float GameTime, float CooldownSeconds);

private:
    void DecayFlags(float GameTime);
    void AggregateFlags(float GameTime);
    void AuthorizeEvents(float GameTime);

    UPROPERTY()
    TArray<FInspectorFlagRecord> FlagHistory;

    UPROPERTY()
    TMap<EInspectorFlag, float> AggregatedScores;

    UPROPERTY()
    TArray<FAuthorizedEventRecord> AuthorizedEvents;

    UPROPERTY(EditDefaultsOnly)
    float FlagWindowSeconds = 72.f * 3600.f;

    UPROPERTY(EditDefaultsOnly)
    float FlagDecayHalfLifeSeconds = 12.f * 3600.f;
};
