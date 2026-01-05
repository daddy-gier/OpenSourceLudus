#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "NHTypes.h"
#include "NHDebugWidget.generated.h"

UCLASS()
class NYGHTSHADEHOLLOWCORE_API UNHDebugWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "Debug")
    void SetObservedActor(AActor* InActor);

    UPROPERTY(BlueprintReadOnly, Category = "Debug")
    FString DebugTimeText;

    UPROPERTY(BlueprintReadOnly, Category = "Debug")
    FString DebugActivityText;

    UPROPERTY(BlueprintReadOnly, Category = "Debug")
    int32 DebugDC;

protected:
    virtual void NativeTick(const FGeometry& MyGeometry, float InDeltaTime) override;

private:
    UPROPERTY()
    TWeakObjectPtr<AActor> ObservedActor;
};
