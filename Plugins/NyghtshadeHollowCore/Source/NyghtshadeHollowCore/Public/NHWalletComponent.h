#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NHWalletComponent.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOWCORE_API UNHWalletComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UNHWalletComponent();

    UFUNCTION(BlueprintCallable, Category = "Wallet")
    void AddDC(int32 Amount);

    UFUNCTION(BlueprintCallable, Category = "Wallet")
    bool SpendDC(int32 Amount);

    UFUNCTION(BlueprintCallable, Category = "Wallet")
    int32 GetDC() const;

private:
    UPROPERTY(EditAnywhere, Category = "Wallet")
    int32 DC;
};
