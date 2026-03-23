#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ViolationRecord.h"
#include "InmateAuthorityComponent.generated.h"

UCLASS(ClassGroup=(Authority), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UInmateAuthorityComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UInmateAuthorityComponent();

    UPROPERTY(BlueprintReadOnly, Replicated)
    int32 AuthorityLevel;

    UPROPERTY(BlueprintReadOnly, Replicated)
    TArray<FViolationRecord> ViolationHistory;

    UFUNCTION(BlueprintCallable)
    void RegisterViolation(
        EViolationType Type,
        int32 Severity,
        FName ReportingGuard,
        const FString& Context
    );

    UFUNCTION(BlueprintCallable)
    void DecayAuthority(float DeltaGameTime);

protected:
    virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& Out) const override;
};
