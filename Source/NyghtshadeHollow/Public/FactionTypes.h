#pragma once

#include "FactionTypes.generated.h"

UENUM(BlueprintType)
enum class EFaction : uint8
{
    Neutral,
    CellBlock,
    YardCrew,
    WorkDetail,
    Guards,
    Administration
};
