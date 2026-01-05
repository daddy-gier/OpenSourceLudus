#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InspectorTypes.h"
#include "CorruptionComponent.generated.h"

UCLASS(ClassGroup=(Authority), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UCorruptionComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UPROPERTY(BlueprintReadOnly, Replicated)
    float CorruptionLevel = 0.f;

    UPROPERTY(BlueprintReadOnly)
    TArray<FBribeRecord> BribeHistory;

    UPROPERTY(BlueprintReadOnly)
    TArray<FAuditFlag> AuditFlags;

    UFUNCTION(BlueprintCallable)
    void AddCorruption(float Amount);

    UFUNCTION(BlueprintCallable)
    bool IsCompromised() const;

    UFUNCTION(BlueprintCallable)
    void RegisterBribe(const FBribeRecord& Record);

    UFUNCTION(BlueprintCallable)
    void RegisterAuditFlag(const FString& Reason, float GameTime);

protected:
    virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& Out) const override;
};
