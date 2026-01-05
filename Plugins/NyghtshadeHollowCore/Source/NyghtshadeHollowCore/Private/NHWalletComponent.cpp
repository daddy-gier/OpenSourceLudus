#include "NHWalletComponent.h"

UNHWalletComponent::UNHWalletComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    DC = 0;
}

void UNHWalletComponent::AddDC(int32 Amount)
{
    if (Amount <= 0)
    {
        return;
    }
    DC += Amount;
}

bool UNHWalletComponent::SpendDC(int32 Amount)
{
    if (Amount <= 0 || Amount > DC)
    {
        return false;
    }
    DC -= Amount;
    return true;
}

int32 UNHWalletComponent::GetDC() const
{
    return DC;
}
