#include "narrative/chapter_i_events.h"

namespace osl::narrative {

ChapterIEvent::ChapterIEvent(FNarrativeEventDefinition definition) : definition_(definition) {}

const FNarrativeEventDefinition& ChapterIEvent::Definition() const {
    return definition_;
}

UnannouncedCellSweepEvent::UnannouncedCellSweepEvent()
    : ChapterIEvent({
          ENarrativeEventId::UnannouncedCellSweep,
          {EInspectorFlag::GuardCollusion},
          3.0f,
          48.0f,
      }) {}

bool UnannouncedCellSweepEvent::CanAuthorize(const FEventContext& context) const {
    auto guard_flag = context.ActiveFlags.find(EInspectorFlag::GuardCollusion);
    bool guard_collusion_flagged = guard_flag != context.ActiveFlags.end() && guard_flag->second;

    return context.ViolenceSpike >= definition_.RequiredIntensity && guard_collusion_flagged &&
           context.InformantConfidence < 0.6f;
}

void UnannouncedCellSweepEvent::ApplyPayload(const FEventContext&, FEventState& state) const {
    state.LockdownActive = true;
    state.ScheduleCompressionActive = true;
    state.GuardAuthorityModifier = 1.15f;
    state.InformantExposureRisk += 0.2f;
}

FEventResidue UnannouncedCellSweepEvent::RegisterResidue(const FEventContext&) const {
    return {
        10.0f,
        5.0f,
        0.05f,
        0.0f,
        0.0f,
    };
}

TransferReviewInitiatedEvent::TransferReviewInitiatedEvent()
    : ChapterIEvent({
          ENarrativeEventId::TransferReviewInitiated,
          {EInspectorFlag::ParoleAbuse},
          2.0f,
          72.0f,
      }) {}

bool TransferReviewInitiatedEvent::CanAuthorize(const FEventContext& context) const {
    auto parole_flag = context.ActiveFlags.find(EInspectorFlag::ParoleAbuse);
    bool parole_abuse_flagged = parole_flag != context.ActiveFlags.end() && parole_flag->second;

    return context.ExternalOversightMonitoring && parole_abuse_flagged &&
           context.JudicialInconsistency >= definition_.RequiredIntensity;
}

void TransferReviewInitiatedEvent::ApplyPayload(const FEventContext&, FEventState& state) const {
    state.OverridesRequireJustification = true;
    state.GuardAuthorityModifier = 1.05f;
}

FEventResidue TransferReviewInitiatedEvent::RegisterResidue(const FEventContext&) const {
    return {
        4.0f,
        0.0f,
        0.0f,
        0.0f,
        0.1f,
    };
}

InformantPressureSpikeEvent::InformantPressureSpikeEvent()
    : ChapterIEvent({
          ENarrativeEventId::InformantPressureSpike,
          {EInspectorFlag::AuditRisk},
          0.7f,
          24.0f,
      }) {}

bool InformantPressureSpikeEvent::CanAuthorize(const FEventContext& context) const {
    return context.InformantConfidence <= 0.5f && context.YardHostility >= 0.7f;
}

void InformantPressureSpikeEvent::ApplyPayload(const FEventContext&, FEventState& state) const {
    state.InformantExposureRisk += 0.3f;
    state.IntelReliabilityVariance += 0.2f;
}

FEventResidue InformantPressureSpikeEvent::RegisterResidue(const FEventContext&) const {
    return {
        6.0f,
        -3.0f,
        0.08f,
        -0.15f,
        0.0f,
    };
}

AdministrativeAuditRequestEvent::AdministrativeAuditRequestEvent()
    : ChapterIEvent({
          ENarrativeEventId::AdministrativeAuditRequest,
          {EInspectorFlag::AuditRisk, EInspectorFlag::GuardCollusion},
          1.0f,
          168.0f,
      }) {}

bool AdministrativeAuditRequestEvent::CanAuthorize(const FEventContext& context) const {
    auto audit_flag = context.ActiveFlags.find(EInspectorFlag::AuditRisk);
    bool audit_flagged = audit_flag != context.ActiveFlags.end() && audit_flag->second;

    auto collusion_flag = context.ActiveFlags.find(EInspectorFlag::GuardCollusion);
    bool collusion_flagged = collusion_flag != context.ActiveFlags.end() && collusion_flag->second;

    return audit_flagged && collusion_flagged && context.AuditRisk >= definition_.RequiredIntensity;
}

void AdministrativeAuditRequestEvent::ApplyPayload(const FEventContext&, FEventState& state) const {
    state.AdminUiRestricted = true;
    state.OverridesRequireJustification = true;
}

FEventResidue AdministrativeAuditRequestEvent::RegisterResidue(const FEventContext&) const {
    return {
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.2f,
    };
}

}  // namespace osl::narrative
