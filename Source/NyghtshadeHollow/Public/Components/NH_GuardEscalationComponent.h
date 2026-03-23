#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Authority/NH_AuthorityTypes.h"
#include "NH_GuardEscalationComponent.generated.h"

class ACharacter;
class UNH_AuthorityComponent;

UENUM(BlueprintType)
enum class ENHGuardResponse : uint8
{
    None UMETA(DisplayName = "None"),
    VerbalWarning UMETA(DisplayName = "Verbal Warning"),
    Supervision UMETA(DisplayName = "Supervision"),
    Restraint UMETA(DisplayName = "Restraint"),
    Segregation UMETA(DisplayName = "Segregation")
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FNHGuardResponseIssued, ACharacter*, Inmate, ENHGuardResponse, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FNHViolationLogged, ACharacter*, Inmate, const FNHViolationRecord&, Violation);

UCLASS(ClassGroup=(Nyghtshade), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UNH_GuardEscalationComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UNH_GuardEscalationComponent();

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Guard Personality")
    int32 AuthorityThreshold = 2;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Guard Personality")
    float MemoryHalfLifeMinutes = 30.0f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Guard Personality")
    bool bPrefersCompliance = true;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Guard Escalation")
    float WarningGraceSeconds = 10.0f;

    UPROPERTY(BlueprintReadOnly, Category="Guard Escalation")
    TWeakObjectPtr<ACharacter> CurrentTarget;

    UPROPERTY(BlueprintAssignable, Category="Guard Escalation")
    FNHGuardResponseIssued OnGuardResponseIssued;

    UPROPERTY(BlueprintAssignable, Category="Guard Escalation")
    FNHViolationLogged OnViolationLogged;

    UFUNCTION(BlueprintCallable, Category="Guard Escalation")
    void EvaluateInmate(ACharacter* Inmate, ENHViolationType ViolationType, int32 Severity, const FString& Context);

    UFUNCTION(BlueprintCallable, Category="Guard Escalation")
    ENHGuardResponse DetermineResponse(const UNH_AuthorityComponent* AuthorityComponent, int32 Severity) const;

protected:
    virtual void BeginPlay() override;

private:
    void ApplyResponse(ACharacter* Inmate, UNH_AuthorityComponent* AuthorityComponent, ENHGuardResponse Response);
};
