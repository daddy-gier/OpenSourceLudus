#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NH_GameRules.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnCrimeReported, AActor*, Perpetrator, int32, Severity);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnLockdownChanged, bool, bIsLockdownActive);

UCLASS()
class NYGHTSHADEHOLLOW_API ANH_GameRules : public AActor
{
    GENERATED_BODY()

public:
    ANH_GameRules();

    UPROPERTY(BlueprintAssignable, Category = "Rules")
    FOnCrimeReported OnCrimeReported;

    UPROPERTY(BlueprintAssignable, Category = "Rules")
    FOnLockdownChanged OnLockdownChanged;

    UFUNCTION(BlueprintCallable, Category = "Rules")
    void NotifyCrime(AActor* Perpetrator, int32 Severity);

    UFUNCTION(BlueprintCallable, Category = "Rules")
    void StartLockdown();

    UFUNCTION(BlueprintCallable, Category = "Rules")
    void EndLockdown();

    UFUNCTION(BlueprintCallable, Category = "Rules")
    bool IsLockdownActive() const;

private:
    bool bLockdownActive = false;
};
