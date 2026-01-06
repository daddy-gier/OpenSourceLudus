#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "ActivityInterface.generated.h"

UINTERFACE(Blueprintable)
class UActivityInterface : public UInterface
{
    GENERATED_BODY()
};

class NYGHTSHADEHOLLOW_API IActivityInterface
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Activity")
    void OnActivityStarted(FName ActivityName);

    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = "Activity")
    void OnActivityEnded(FName ActivityName);
};
