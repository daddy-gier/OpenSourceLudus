using UnrealBuildTool;

public class NyghtshadeHollow : ModuleRules
{
    public NyghtshadeHollow(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "AIModule",
            "GameplayTasks",
            "NavigationSystem",
            "NetCore"
        });
    }
}
