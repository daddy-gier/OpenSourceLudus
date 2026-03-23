# RevoltGPT Unreal Usage Notes

## Blueprint UI Example

```
Event OnClicked (SendButton)
-> GetText(PromptTextBox) -> Local Prompt
-> GenerateTextAsync(Prompt = Prompt, Callback = Event_OnRevoltResponse)

Event_OnRevoltResponse(bool bSuccess, string Response)
-> Branch (bSuccess)
True:
-> SetText(ResponseText, Response)
-> Print String Response
-> (Optional) Save string to file
False:
-> SetText(ResponseText, "Error: " + Response)
-> Print String("Revolt request failed: " + Response)
```

## Prompt Template (AI Assistant)

```
You are an expert Unreal C++/Blueprint engineer.
Project: Nyghtshade Hollow (UE5).
File: {relative_path}
Function/Blueprint node: {context}
Constraint: {constraints}
Goal: {goal}
Provide: 1) patch/diff 2) short explanation of why this is safe 3) tests to run.
```

## Project Context Example

```
Context: [project metadata]
File: /Source/MyGame/Player.cpp
Function: void APlayer::Tick(float Delta)
Constraint: Do not modify network replication logic. Keep performance below X.
Task: Implement sprint toggle based on input and stamina meter.
```
