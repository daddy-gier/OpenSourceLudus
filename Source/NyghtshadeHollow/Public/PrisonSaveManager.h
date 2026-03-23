#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PrisonSaveManager.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnPrisonSaved);

UCLASS()
class NYGHTSHADEHOLLOW_API APrisonSaveManager : public AActor
{
    GENERATED_BODY()

public:
    APrisonSaveManager();

    UPROPERTY(BlueprintAssignable)
    FOnPrisonSaved OnPrisonSaved;

    UFUNCTION(BlueprintCallable)
    void SavePrison();

    UFUNCTION(BlueprintCallable)
    void LoadPrison();

protected:
    UPROPERTY(EditDefaultsOnly)
    FString SaveSlot = TEXT("PrisonSlot_01");
};
