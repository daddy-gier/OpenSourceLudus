#include "Modules/ModuleManager.h"

class FNyghtshadeHollowCoreModule : public IModuleInterface
{
public:
    virtual void StartupModule() override {}
    virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FNyghtshadeHollowCoreModule, NyghtshadeHollowCore)
