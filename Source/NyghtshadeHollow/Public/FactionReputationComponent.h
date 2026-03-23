#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "FactionTypes.h"
#include "FactionReputationComponent.generated.h"

UCLASS(ClassGroup=(Authority), meta=(BlueprintSpawnableComponent))
class NYGHTSHADEHOLLOW_API UFactionReputationComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UPROPERTY(BlueprintReadWrite)
    TMap<EFaction, float> FactionReputation;
};
