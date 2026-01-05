#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Authority/NH_AuthorityTypes.h"
#include "NH_AuthorityComponent.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FNHAuthorityLevelChanged, ENHAuthorityLevel, PreviousLevel, ENHAuthorityLevel, NewLevel);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FNHPunishmentChanged, const FNHPunishmentRecord&, Punishment);

UCLASS(ClassGroup=(Nyghtshade), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UNH_AuthorityComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UNH_AuthorityComponent();

    UPROPERTY(BlueprintReadOnly, Category="Authority")
    ENHAuthorityLevel CurrentAuthorityLevel = ENHAuthorityLevel::Free;

    UPROPERTY(BlueprintReadOnly, Category="Authority")
    TArray<FNHViolationRecord> ViolationHistory;

    UPROPERTY(BlueprintReadOnly, Category="Authority")
    TArray<FNHPunishmentRecord> ActivePunishments;

    UPROPERTY(BlueprintAssignable, Category="Authority")
    FNHAuthorityLevelChanged OnAuthorityLevelChanged;

    UPROPERTY(BlueprintAssignable, Category="Authority")
    FNHPunishmentChanged OnPunishmentStarted;

    UPROPERTY(BlueprintAssignable, Category="Authority")
    FNHPunishmentChanged OnPunishmentEnded;

    UFUNCTION(BlueprintCallable, Category="Authority")
    void AddViolation(const FNHViolationRecord& Violation, bool bAutoEscalate);

    UFUNCTION(BlueprintCallable, Category="Authority")
    void SetAuthorityLevel(ENHAuthorityLevel NewLevel);

    UFUNCTION(BlueprintCallable, Category="Authority")
    void BeginPunishment(const FNHPunishmentRecord& NewPunishment);

    UFUNCTION(BlueprintCallable, Category="Authority")
    void EndPunishmentByType(FName PunishmentType);

    UFUNCTION(BlueprintCallable, Category="Authority")
    bool HasActivePunishment(FName PunishmentType) const;

    UFUNCTION(BlueprintCallable, Category="Authority")
    FNHPunishmentRecord GetActivePunishment(FName PunishmentType, bool& bFound) const;

    UFUNCTION(BlueprintCallable, Category="Authority")
    void TickPunishments(float DeltaSeconds);

protected:
    virtual void BeginPlay() override;
};
