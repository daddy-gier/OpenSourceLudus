#include "RevoltGPTSecureBPLibrary.h"

#include "RevoltGPTSecure.h"

void URevoltGPTSecureBPLibrary::InitRevoltApiKeyFromEnv()
{
  RevoltSecure::InitApiKey();
}
