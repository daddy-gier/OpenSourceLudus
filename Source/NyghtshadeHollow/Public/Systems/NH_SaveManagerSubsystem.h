#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Save/NH_SaveGame.h"
#include "NH_SaveManagerSubsystem.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FNHPrisonSaved, const FString&, SlotName);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FNHPrisonLoaded, const FString&, SlotName);

UCLASS()
class NYGHTSHADEHOLLOW_API UNH_SaveManagerSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Save")
    FString ManualSlotName = TEXT("NyghtshadeManual");

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Save")
    FString AutosaveSlotPrefix = TEXT("NyghtshadeAuto");

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Save")
    int32 AutosaveSlots = 3;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Save")
    float AutosaveIntervalSeconds = 120.0f;

    UPROPERTY(BlueprintAssignable, Category="Save")
    FNHPrisonSaved OnPrisonSaved;

    UPROPERTY(BlueprintAssignable, Category="Save")
    FNHPrisonLoaded OnPrisonLoaded;

    UFUNCTION(BlueprintCallable, Category="Save")
    void StartAutosave();

    UFUNCTION(BlueprintCallable, Category="Save")
    void StopAutosave();

    UFUNCTION(BlueprintCallable, Category="Save")
    void SavePrisonManual();

    UFUNCTION(BlueprintCallable, Category="Save")
    void SavePrisonToSlot(const FString& SlotName);

    UFUNCTION(BlueprintCallable, Category="Save")
    bool LoadPrisonFromSlot(const FString& SlotName);

    UFUNCTION(BlueprintCallable, Category="Save")
    void TriggerRollingAutosave();

protected:
    virtual void Deinitialize() override;

private:
    FTimerHandle AutosaveHandle;
    int32 AutosaveIndex = 0;

    UNH_SaveGame* GatherSaveData() const;
    void ApplySaveData(const UNH_SaveGame* SaveData) const;
};
