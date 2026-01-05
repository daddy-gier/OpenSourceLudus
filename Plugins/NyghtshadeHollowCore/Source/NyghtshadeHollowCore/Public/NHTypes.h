#pragma once

#include "CoreMinimal.h"
#include "Engine/DataTable.h"
#include "NHTypes.generated.h"

UENUM(BlueprintType)
enum class ENHActivityType : uint8
{
    None UMETA(DisplayName = "None"),
    Sleep UMETA(DisplayName = "Sleep"),
    Work UMETA(DisplayName = "Work"),
    Eat UMETA(DisplayName = "Eat"),
    Exercise UMETA(DisplayName = "Exercise"),
    Patrol UMETA(DisplayName = "Patrol"),
    Idle UMETA(DisplayName = "Idle")
};

UENUM(BlueprintType)
enum class ENHFactionId : uint8
{
    None UMETA(DisplayName = "None"),
    Administration UMETA(DisplayName = "Administration"),
    COs UMETA(DisplayName = "COs"),
    Medical UMETA(DisplayName = "Medical"),
    Kitchen UMETA(DisplayName = "Kitchen"),
    Teachers UMETA(DisplayName = "Teachers"),
    Inmates_General UMETA(DisplayName = "Inmates General"),
    Inmates_CrewA UMETA(DisplayName = "Inmates Crew A"),
    Inmates_CrewB UMETA(DisplayName = "Inmates Crew B"),
    Chapel UMETA(DisplayName = "Chapel")
};

UENUM(BlueprintType)
enum class ENHContractType : uint8
{
    Intimidate UMETA(DisplayName = "Intimidate"),
    Steal UMETA(DisplayName = "Steal"),
    Sabotage UMETA(DisplayName = "Sabotage"),
    AssaultGameplayOnly UMETA(DisplayName = "Assault (Gameplay Only)")
};

UENUM(BlueprintType)
enum class ENHContractStatus : uint8
{
    Requested UMETA(DisplayName = "Requested"),
    Assigned UMETA(DisplayName = "Assigned"),
    InProgress UMETA(DisplayName = "In Progress"),
    Completed UMETA(DisplayName = "Completed"),
    Failed UMETA(DisplayName = "Failed")
};

USTRUCT(BlueprintType)
struct FNHActivityRow : public FTableRowBase
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 StartMinute = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 EndMinute = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    ENHActivityType Activity = ENHActivityType::None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName TargetTag = NAME_None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Notes;
};

USTRUCT(BlueprintType)
struct FNHContract
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FGuid Id;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    ENHContractType Type = ENHContractType::Intimidate;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FName TargetIdTag = NAME_None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 PriceDC = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    ENHContractStatus Status = ENHContractStatus::Requested;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float SuccessChanceVisible = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float SuccessChanceHidden = 0.5f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Notes;
};
