#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "PunishmentRecord.h"
#include "InmateJudicialComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FOnPunishmentChanged,
    const FPunishmentRecord&, Punishment
);

UCLASS(ClassGroup=(Authority), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UInmateJudicialComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UInmateJudicialComponent();

    UPROPERTY(BlueprintReadOnly, Replicated)
    TArray<FPunishmentRecord> ActivePunishments;

    UPROPERTY(BlueprintAssignable)
    FOnPunishmentChanged OnPunishmentAdded;

    UFUNCTION(BlueprintCallable)
    void SentenceInmate(
        EPunishmentType Type,
        float Duration,
        const FString& Reason
    );

    UFUNCTION(BlueprintCallable)
    void TickPunishments(float DeltaSeconds);

    UFUNCTION(BlueprintCallable)
    bool IsInSegregation() const;

protected:
    virtual void GetLifetimeReplicatedProps(
        TArray<FLifetimeProperty>& Out
    ) const override;
};
