#pragma once

#include "PunishmentTypes.generated.h"

UENUM(BlueprintType)
enum class EPunishmentType : uint8
{
    Warning,
    PrivilegeLoss,
    RestrictedMovement,
    SolitarySegregation,
    IndefiniteSegregation
};
