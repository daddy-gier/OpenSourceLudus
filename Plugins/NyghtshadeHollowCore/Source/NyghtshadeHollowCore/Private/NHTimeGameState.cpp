#include "NHTimeGameState.h"

ANHTimeGameState::ANHTimeGameState()
{
    PrimaryActorTick.bCanEverTick = true;
    MinutesPerRealSecond = 10.0f;
    CurrentMinuteOfDay = 360;
    CurrentDay = 1;
    bPausedTime = false;
    MinuteAccumulator = 0.0f;
}

void ANHTimeGameState::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    if (bPausedTime || MinutesPerRealSecond <= 0.0f)
    {
        return;
    }

    MinuteAccumulator += DeltaSeconds * MinutesPerRealSecond;
    const int32 MinutesToAdvance = FMath::FloorToInt(MinuteAccumulator);
    if (MinutesToAdvance > 0)
    {
        MinuteAccumulator -= MinutesToAdvance;
        AdvanceMinutes(MinutesToAdvance);
    }
}

void ANHTimeGameState::AdvanceMinutes(int32 MinutesToAdvance)
{
    int32 NewMinute = CurrentMinuteOfDay + MinutesToAdvance;
    while (NewMinute >= 1440)
    {
        NewMinute -= 1440;
        CurrentDay++;
    }

    CurrentMinuteOfDay = NewMinute;
    OnTimeChanged.Broadcast(CurrentDay, CurrentMinuteOfDay);
}

FString ANHTimeGameState::GetTimeHHMM() const
{
    const int32 Hours = CurrentMinuteOfDay / 60;
    const int32 Minutes = CurrentMinuteOfDay % 60;
    return FString::Printf(TEXT("%02d:%02d"), Hours, Minutes);
}

void ANHTimeGameState::SetTimePaused(bool bPaused)
{
    bPausedTime = bPaused;
}

void ANHTimeGameState::SetTimeScale(float MinutesPerSecond)
{
    MinutesPerRealSecond = FMath::Max(0.0f, MinutesPerSecond);
}
