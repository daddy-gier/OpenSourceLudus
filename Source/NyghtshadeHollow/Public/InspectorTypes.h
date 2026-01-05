#pragma once

#include "CoreMinimal.h"
#include "InspectorTypes.generated.h"

UENUM(BlueprintType)
enum class EInspectorFlag : uint8
{
    ViolenceSpike,
    CorruptionTrend,
    ParoleAbuse,
    AuditRisk,
    EvidenceTampering,
    GuardCollusion
};

UENUM(BlueprintType)
enum class EAuthorizedNarrativeEvent : uint8
{
    CellSweep,
    TransferReview,
    InformantOffer,
    ExternalAuditPing
};

USTRUCT(BlueprintType)
struct FInspectorFlagRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    EInspectorFlag Flag = EInspectorFlag::ViolenceSpike;

    UPROPERTY(BlueprintReadOnly)
    float Weight = 0.f;

    UPROPERTY(BlueprintReadOnly)
    float Timestamp = 0.f;

    UPROPERTY(BlueprintReadOnly)
    FString Source;
};

USTRUCT(BlueprintType)
struct FBribeRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FName BribedActor;

    UPROPERTY(BlueprintReadOnly)
    float Amount = 0.f;

    UPROPERTY(BlueprintReadOnly)
    float GameTime = 0.f;

    UPROPERTY(BlueprintReadOnly)
    FString Effect;
};

USTRUCT(BlueprintType)
struct FAppealRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float FiledTime = 0.f;

    UPROPERTY(BlueprintReadOnly)
    FString Reason;

    UPROPERTY(BlueprintReadOnly)
    bool bSuccessful = false;
};

USTRUCT(BlueprintType)
struct FAuditFlag
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float Time = 0.f;

    UPROPERTY(BlueprintReadOnly)
    FString Reason;
};

USTRUCT(BlueprintType)
struct FAuthorizedEventRecord
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    EAuthorizedNarrativeEvent Event = EAuthorizedNarrativeEvent::CellSweep;

    UPROPERTY(BlueprintReadOnly)
    float AuthorizedTime = 0.f;

    UPROPERTY(BlueprintReadOnly)
    float CooldownEndTime = 0.f;
};
