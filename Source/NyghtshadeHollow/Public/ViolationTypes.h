#pragma once

#include "ViolationTypes.generated.h"

UENUM(BlueprintType)
enum class EViolationType : uint8
{
    Schedule,
    DoorAccess,
    RestrictedArea,
    Contraband,
    Disobedience,
    Violence
};
