using UnrealBuildTool;

public class NyghtshadeHollowCore : ModuleRules
{
    public NyghtshadeHollowCore(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "AIModule",
                "GameplayTasks",
                "NavigationSystem",
                "UMG",
                "Slate",
                "SlateCore"
            }
        );
    }
}
