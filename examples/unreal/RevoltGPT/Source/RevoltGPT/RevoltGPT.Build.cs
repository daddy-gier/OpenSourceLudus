using UnrealBuildTool;
using System.Collections.Generic;

public class RevoltGPT : ModuleRules
{
  public RevoltGPT(ReadOnlyTargetRules Target) : base(Target)
  {
    PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

    PublicDependencyModuleNames.AddRange(new string[] {
      "Core",
      "CoreUObject",
      "Engine",
      "InputCore",
      "BlueprintGraph"
    });

    PrivateDependencyModuleNames.AddRange(new string[] {
      "Projects",
      "HTTP",
      "Json",
      "JsonUtilities"
    });
  }
}
