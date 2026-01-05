#include "NHDebugWidget.h"
#include "NHTimeGameState.h"
#include "NHScheduleComponent.h"
#include "NHWalletComponent.h"

void UNHDebugWidget::SetObservedActor(AActor* InActor)
{
    ObservedActor = InActor;
}

void UNHDebugWidget::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
    Super::NativeTick(MyGeometry, InDeltaTime);

    DebugTimeText = TEXT("--:--");
    DebugActivityText = TEXT("None");
    DebugDC = 0;

    if (UWorld* World = GetWorld())
    {
        if (ANHTimeGameState* TimeState = World->GetGameState<ANHTimeGameState>())
        {
            DebugTimeText = TimeState->GetTimeHHMM();
        }
    }

    if (AActor* Actor = ObservedActor.Get())
    {
        if (UNHScheduleComponent* Schedule = Actor->FindComponentByClass<UNHScheduleComponent>())
        {
            const ENHActivityType Activity = Schedule->GetCurrentActivityType();
            DebugActivityText = UEnum::GetDisplayValueAsText(Activity).ToString();
        }

        if (UNHWalletComponent* Wallet = Actor->FindComponentByClass<UNHWalletComponent>())
        {
            DebugDC = Wallet->GetDC();
        }
    }
}
