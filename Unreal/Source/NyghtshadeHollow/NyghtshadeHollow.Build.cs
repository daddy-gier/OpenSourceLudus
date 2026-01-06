using UnrealBuildTool;

public class NyghtshadeHollow : ModuleRules
{
    public NyghtshadeHollow(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "InputCore",
                "AIModule",
                "NavigationSystem",
                "UMG"
            }
        );

        PrivateDependencyModuleNames.AddRange(new string[] { });
    }
}
