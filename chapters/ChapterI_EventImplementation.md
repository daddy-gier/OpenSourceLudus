# Chapter I — Event Implementation

Tone: neutral, procedural, canonical.

This document implements Chapter I event execution and the downstream escalation systems requested. All definitions are system-bound and persist across saves. No sarcasm, no commentary.

## 1. Event Implementation Principles

* Events are constraint injections, not story beats.
* Events are authorized externally (Inspector or Oversight), executed internally by systems.
* Every event leaves persistent residue.
* Events never resolve themselves unless explicitly reversible.

## 2. Event Execution Pipeline

1. **Authorization**: `InspectorAuthoritySubsystem` evaluates conditions and cooldowns.
2. **Instantiation**: Event is marked active and logged in save data by `EventId`.
3. **Payload**: Deterministic constraints apply to existing systems.
4. **Residue**: Long-term modifiers are recorded.
5. **Monitoring**: Effects persist until superseded or explicitly reversed.

## 3. Base Event Interface

```
class UChapterIEvent : public UObject
{
    GENERATED_BODY()

public:
    FName EventId;

    virtual bool CanAuthorize() const;
    virtual void ApplyPayload();
    virtual void RegisterResidue();
};
```

## 4. Implemented Chapter I Events

### A. Unannounced Cell Sweep

**Authorization**

* ViolenceSpike ≥ threshold
* GuardCollusion flagged
* Informant confidence degraded

**Payload**

* `ScheduleManager` enforces schedule compression
* Lockdown doors engage (staggered)
* Guard authority modifier applied
* Informant `ExposureRisk` increases

**Residue**

* Faction hostility increases
* Inspector confidence modifier increases
* False-positive violation rate increases

### B. Transfer Review Initiated

**Authorization**

* Judicial inconsistencies detected
* ParoleAbuse flagged repeatedly
* `ExternalOversightSubsystem` is Monitoring or Active

**Payload**

* Selected inmates flagged “Under Review”
* Schedule overrides restricted
* Guard escort frequency increases
* Appeals weighted negatively

**Residue**

* Inmate paranoia increases
* Informant recruitment likelihood rises
* Judicial discretion reduced

### C. Informant Pressure Spike

**Authorization**

* Active informant network present
* `ExposureRisk` trending upward
* Yard hostility elevated

**Payload**

* Informant timers accelerate
* Intel reliability variance increases
* Retaliation micro-events enabled
* Guard response ambiguity increases

**Residue**

* Informant trust ceiling lowered
* Inspector intel confidence recalibrated
* Long-term data noise introduced

### D. Administrative Audit Request

**Authorization**

* AuditRisk critical
* GuardCollusion unresolved
* Punishment metrics inconsistent

**Payload**

* Oversight escalation check triggered
* Admin UI permissions reduced
* Overrides generate immediate flags
* Save metadata marked “Under Audit”

**Residue**

* Future admin actions weighted higher
* Audit presence persists across saves
* Recovery difficulty increased

## 5. Event Interaction Rules

* Multiple Chapter I events can coexist.
* Effects stack multiplicatively where defined.
* Residue compounds across events.
* Events never cancel each other.

## 6. Save/Load Behavior

* Event ID, authorization timestamp, residue state, and cooldown are serialized.
* Reloading restores full event context.
* Events cannot be avoided by reload.

---

# Investigation Escalation Events

## 1. Core Purpose

External investigations remove ambiguity from records, reduce local autonomy, and force irreversible outcomes when correction fails. Intent and internal politics are ignored.

## 2. Investigation State Machine

```
UENUM(BlueprintType)
enum class EInvestigationStage : uint8
{
    Dormant,
    Monitoring,
    Active,
    Embedded,
    Intervention,
    Resolution
};
```

The current stage is owned by `ExternalOversightSubsystem` and persisted across saves. Stages do not regress without resolution.

## 3. Stage Definitions

### Stage I — Monitoring

**Entry Conditions**

* AuditRisk sustained above threshold
* ≥2 unresolved Inspector flags
* No recent corrective outcomes

**System Effects**

* Increased logging granularity
* Inspector flag decay slowed
* Admin overrides tagged but allowed

### Stage II — Active Investigation

**Entry Conditions**

* Monitoring duration exceeded
* Continued contradiction between reports and outcomes
* Informant or guard suppression detected

**System Effects**

* Audit requests generated automatically
* Punishment justifications required
* Overrides logged with higher weight
* Certain actions raise AuditRisk immediately

### Stage III — Embedded Oversight

**Entry Conditions**

* Active stage unresolved
* GuardCollusion or EvidenceTampering persists
* Cascade events detected

**System Effects**

* Oversight agent instantiated (NPC or abstract)
* Random audits triggered
* Admin UI permissions reduced
* Informant handling monitored
* Guard reassignment probability increased

### Stage IV — Intervention

**Entry Conditions**

* Embedded oversight ineffective
* Systemic Breakpoint reached
* Multiple cascade events unresolved

**System Effects**

* External authority overrides local decisions
* Forced transfers initiated
* Segregation durations capped or recalculated
* Guard leadership reshuffled
* Admin cannot suppress flags

### Stage V — Resolution

**Entry Conditions**

* External authority concludes outcome
* Time-based or evidence-based conclusion

**Possible Outcomes**

* Clearance with permanent audit markers
* Structural sanctions
* Leadership replacement
* Facility downgrade
* Persistent oversight flag in save data

## 4. Escalation Triggers (Deterministic)

Examples:

* Repeated overrides without outcome change
* Informant exposure rates exceed intel yield
* Punishment severity variance outside norms
* Guard integrity decay without corrective action

All triggers are logged and replayable.

## 5. Integration Points

* `ScheduleManager`: Receives authority reduction flags; forced overrides may occur.
* Judicial system: Sentencing discretion reduced; appeals weighted externally.
* Save system: Investigation stage serialized; reload does not reset stage.
* `InspectorAuthoritySubsystem`: Flag weighting increased; confidence thresholds tightened.

---

# Role-Transition Events

Role transitions are authorized state changes driven by records, thresholds, and oversight outcomes. They are save-bound, audited, and non-cosmetic.

## 1. Role Definitions

```
UENUM(BlueprintType)
enum class EPlayerRole : uint8
{
    Inmate,
    Administrator
};
```

## 2. Authorization Sources

Role transitions may only be authorized by:

* `ExternalOversightSubsystem`
* `InspectorAuthoritySubsystem`
* Investigation Stage IV or V resolution
* Predefined Chapter I event definitions

No manual switching. No menu access.

## 3. Role-Transition Event Structure

```
struct FRoleTransitionEvent
{
    FName EventId;
    EPlayerRole FromRole;
    EPlayerRole ToRole;

    TArray<EInspectorFlag> RequiredFlags;
    EInvestigationStage MinimumInvestigationStage;

    bool bReversible;
    float CooldownHours;

    void ApplyRoleChange();
};
```

## 4. Transition Types

### A. Inmate → Administrator

**Authorization Examples**

* Oversight Resolution requires internal compliance proxy
* Player integrity metrics acceptable
* No active violent violations at authorization time

**Effects**

* Admin UI access granted
* Inspector/Oversight data visible
* Former inmate records remain immutable
* Faction hostility recalculated
* Initial permissions scoped and logged

### B. Administrator → Inmate

**Authorization Examples**

* Oversight Intervention
* EvidenceTampering confirmed
* GuardCollusion unresolved

**Effects**

* Admin UI revoked
* Player reassigned to inmate entity
* Punishment/sentence recalculated
* Former admin actions persist in records

### C. Administrator → Observer (Optional)

**Authorization Examples**

* Oversight embedded
* Player removed from operational authority

**Effects**

* Read-only Inspector/Oversight access
* No override permissions

## 5. Permission Rebinding

On transition, systems rebind:

* UI layers
* Input permissions
* Action authorization gates
* Visibility scopes

No system logic is rewritten. Only access gates change.

## 6. Save/Load Behavior

* Role state serialized in save data
* Transition event ID recorded
* Reversal only if `bReversible == true`
* Reloading does not bypass authorization

---

# Systemic Collapse Simulation

## A. Entry Conditions

Collapse simulation begins when all are true:

* ≥3 active corruption cascade events
* Investigation stage ≥ Embedded
* Informant network degraded or eliminated
* Guard integrity variance exceeds tolerance
* Admin intervention frequency above threshold
* Inspector confidence in correction below minimum

## B. Collapse States (Ordered)

1. **Governance Drift**: schedules desynchronize; enforcement inconsistent.
2. **Enforcement Fragmentation**: guards act locally; punishments lose proportionality.
3. **Data Integrity Failure**: flags contradict outcomes; audits diverge.
4. **Authority Paralysis**: overrides disabled/ignored; decision latency increases.
5. **Systemic Breakpoint**: external authority assumes control.

## C. Collapse Residue (Permanent)

* Save file flagged `bPostSystemicCollapse = true`
* Recovery paths limited
* Oversight presence persistent
* Reputation and trust metrics capped

---

# Oversight Stress Test

## Inputs

Injected over simulated time:

* High-frequency violations
* Conflicting informant intel
* Guard collusion clusters
* Administrative suppression attempts
* Judicial inconsistency

## Validation Criteria

Oversight must:

* Advance stages without skipping
* Never regress stages
* Persist state across saves
* Ignore player intent
* Prioritize record consistency

Failure if:

* Oversight stalls
* Authority can override escalation
* Reload alters stage
* Resolution clears without residue

---

# Irreversible Role Transition Scenarios

## Scenario I — Admin Removal

**Authorization**

* Oversight Intervention reached
* EvidenceTampering or GuardCollusion confirmed
* Collapse state ≥ Breakpoint

**Execution**

* Administrator → Inmate
* Admin UI permanently revoked
* Former actions remain visible in records

**Irreversibility**

* No appeal
* Save-bound

## Scenario II — Conditional Admin Appointment

**Authorization**

* Oversight Resolution requires internal proxy
* Player integrity metrics acceptable
* Collapse stabilized but unresolved

**Execution**

* Inmate → Administrator
* Permissions scoped and monitored
* AuditRisk elevated permanently

## Scenario III — Observer Lockout

**Authorization**

* Oversight Embedded
* Player deemed unfit for operational control

**Execution**

* Read-only Inspector/Oversight access
* No operational control

---

# Post-Collapse Narrative Arcs

## Arc A — Institutional Normalization

**Conditions**

* Collapse resolved via Intervention/Resolution
* Oversight remains embedded
* No further cascades active

**Characteristics**

* Daily life resumes with increased rigidity
* Punishments consistent but harsh
* Appeals exist but rarely succeed

## Arc B — Lingering Rot

**Conditions**

* Collapse reached Breakpoint
* Oversight reduced but not removed
* Residual corruption persists

**Characteristics**

* Factions regain influence
* Informants unreliable
* Punishments drift subtly

## Arc C — Administrative Purge

**Conditions**

* Oversight Resolution with leadership removal
* Admin displaced

**Characteristics**

* Staff turnover spikes
* Policies rewritten
* New inconsistencies emerge

## Arc D — Total Oversight Dominance

**Conditions**

* Persistent audit flags
* Multi-facility oversight active
* Local governance dissolved

**Characteristics**

* Zero discretion
* Algorithmic punishment
* No informal power structures

---

# New Game+ Inheritance Rules

## Inherited Data (Mandatory)

* Oversight stage
* Collapse flag (`bPostSystemicCollapse`)
* Permanent role transitions
* Reputation caps
* Faction memory
* Audit markers

## Role Inheritance

* If player ended as inmate → may start as different inmate
* If player ended as admin → authority may be reduced or denied
* Observer lockout persists unless explicitly resolved

## System Modifiers

* Higher baseline scrutiny
* Reduced tolerance thresholds
* Faster escalation curves
* Earlier oversight engagement

## Content Variation

* Certain arcs unavailable post-collapse
* Some investigations trigger earlier
* Informant networks less stable
* Appeals less effective

---

# Multi-Facility Oversight Expansion

## Expansion Triggers

* Repeated collapses across saves
* Persistent audit anomalies
* Oversight confidence in local reform below threshold

## System Effects

* Oversight policies standardized
* Cross-facility benchmarks applied
* Transfers influenced by system-wide metrics
* Local variance penalized

## Data Sharing

* Records anonymized and aggregated
* Patterns detected across facilities
* Individual actions weighed against system norms
## Player Impact

* Local success insufficient
* Systemic compliance required
* Individual reform less meaningful

---

# Canon Rules (Locked)

* Events are constraint injections.
* Investigations are stage-driven and persistent.
* Role transitions are record-driven and audited.
* Collapse states persist across saves.
* Oversight escalation is irreversible.
* Authority never erases history.
* The prison does not depend on the player.
