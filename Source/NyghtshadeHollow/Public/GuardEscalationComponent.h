#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ViolationTypes.h"
#include "GuardEscalationComponent.generated.h"

UCLASS(ClassGroup=(Authority), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UGuardEscalationComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere)
    int32 AuthorityThreshold = 2;

    UPROPERTY(EditAnywhere)
    float MemoryHalfLife = 1800.f;

    UFUNCTION(BlueprintCallable)
    void EvaluateInmate(ACharacter* Inmate);

protected:
    void IssueWarning(ACharacter* Inmate);
    void Escalate(ACharacter* Inmate, int32 CurrentLevel);
};
