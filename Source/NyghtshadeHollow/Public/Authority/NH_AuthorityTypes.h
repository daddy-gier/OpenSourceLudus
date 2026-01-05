#pragma once

#include "CoreMinimal.h"
#include "NH_AuthorityTypes.generated.h"

UENUM(BlueprintType)
enum class ENHAuthorityLevel : uint8
{
    Free UMETA(DisplayName = "Free"),
    VerballyWarned UMETA(DisplayName = "Verbally Warned"),
    UnderSupervision UMETA(DisplayName = "Under Supervision"),
    Restrained UMETA(DisplayName = "Restrained / Escorted"),
    Segregated UMETA(DisplayName = "Segregated"),
    InstitutionalRisk UMETA(DisplayName = "Institutional Risk")
};

UENUM(BlueprintType)
enum class ENHViolationType : uint8
{
    Schedule UMETA(DisplayName = "Schedule"),
    DoorAccess UMETA(DisplayName = "Door Access"),
    RestrictedArea UMETA(DisplayName = "Restricted Area"),
    Contraband UMETA(DisplayName = "Contraband"),
    Disobedience UMETA(DisplayName = "Disobedience"),
    Violence UMETA(DisplayName = "Violence")
};

USTRUCT(BlueprintType)
struct FNHViolationRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    ENHViolationType Type = ENHViolationType::Schedule;

    UPROPERTY(BlueprintReadOnly)
    int32 Severity = 1;

    UPROPERTY(BlueprintReadOnly)
    float GameTimeStamp = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    FName ReportingGuard;

    UPROPERTY(BlueprintReadOnly)
    FString Context;
};

USTRUCT(BlueprintType)
struct FNHPunishmentRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FName PunishmentType;

    UPROPERTY(BlueprintReadOnly)
    float StartTime = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float Duration = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float TimeServed = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    bool bParoleEligible = false;

    UPROPERTY(BlueprintReadOnly)
    bool bIsActive = true;
};
