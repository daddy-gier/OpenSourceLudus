#ifndef OPEN_SOURCE_LUDUS_CHAPTER_I_EVENTS_H
#define OPEN_SOURCE_LUDUS_CHAPTER_I_EVENTS_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace osl::narrative {

enum class EInspectorFlag : uint8_t {
    None,
    GuardCollusion,
    ParoleAbuse,
    EvidenceTampering,
    AuditRisk,
};

enum class ENarrativeEventId : uint8_t {
    UnannouncedCellSweep,
    TransferReviewInitiated,
    InformantPressureSpike,
    AdministrativeAuditRequest,
};

struct FNarrativeEventDefinition {
    ENarrativeEventId EventId;
    std::vector<EInspectorFlag> RequiredFlags;
    float RequiredIntensity = 0.0f;
    float CooldownHours = 0.0f;
};

struct FEventResidue {
    float FactionHostilityDelta = 0.0f;
    float InspectorConfidenceDelta = 0.0f;
    float FalsePositiveRateDelta = 0.0f;
    float InformantTrustCeilingDelta = 0.0f;
    float AuditPressureDelta = 0.0f;
};

struct FEventContext {
    float ViolenceSpike = 0.0f;
    float GuardCollusionScore = 0.0f;
    float InformantConfidence = 1.0f;
    float AuditRisk = 0.0f;
    float YardHostility = 0.0f;
    float JudicialInconsistency = 0.0f;
    bool ExternalOversightMonitoring = false;

    std::unordered_map<EInspectorFlag, bool> ActiveFlags;
};

struct FEventState {
    bool LockdownActive = false;
    bool ScheduleCompressionActive = false;
    bool AdminUiRestricted = false;
    bool OverridesRequireJustification = false;
    float GuardAuthorityModifier = 1.0f;
    float InformantExposureRisk = 0.0f;
    float IntelReliabilityVariance = 0.0f;
};

class ChapterIEvent {
public:
    explicit ChapterIEvent(FNarrativeEventDefinition definition);
    virtual ~ChapterIEvent() = default;

    const FNarrativeEventDefinition& Definition() const;

    virtual bool CanAuthorize(const FEventContext& context) const = 0;
    virtual void ApplyPayload(const FEventContext& context, FEventState& state) const = 0;
    virtual FEventResidue RegisterResidue(const FEventContext& context) const = 0;

protected:
    FNarrativeEventDefinition definition_;
};

class UnannouncedCellSweepEvent final : public ChapterIEvent {
public:
    UnannouncedCellSweepEvent();

    bool CanAuthorize(const FEventContext& context) const override;
    void ApplyPayload(const FEventContext& context, FEventState& state) const override;
    FEventResidue RegisterResidue(const FEventContext& context) const override;
};

class TransferReviewInitiatedEvent final : public ChapterIEvent {
public:
    TransferReviewInitiatedEvent();

    bool CanAuthorize(const FEventContext& context) const override;
    void ApplyPayload(const FEventContext& context, FEventState& state) const override;
    FEventResidue RegisterResidue(const FEventContext& context) const override;
};

class InformantPressureSpikeEvent final : public ChapterIEvent {
public:
    InformantPressureSpikeEvent();

    bool CanAuthorize(const FEventContext& context) const override;
    void ApplyPayload(const FEventContext& context, FEventState& state) const override;
    FEventResidue RegisterResidue(const FEventContext& context) const override;
};

class AdministrativeAuditRequestEvent final : public ChapterIEvent {
public:
    AdministrativeAuditRequestEvent();

    bool CanAuthorize(const FEventContext& context) const override;
    void ApplyPayload(const FEventContext& context, FEventState& state) const override;
    FEventResidue RegisterResidue(const FEventContext& context) const override;
};

}  // namespace osl::narrative

#endif  // OPEN_SOURCE_LUDUS_CHAPTER_I_EVENTS_H
