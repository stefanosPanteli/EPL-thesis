# EPL-thesis
Multi-agent system for the thesis.

`author:` Stefanos Panteli

# Agents

The following agents are used in the system:

## Task Decomposer
***Purpose:*** Analyzes the request and determines whether a single agent or multiple specialized agents are needed.

***Input:*** 
```json
{
    "user_input": "..."
}
```
***Output:*** 
```json
{
    "user_input": "...",
    "refined_input": "...",
    "output": {
        "type": "multi-agent", // or "single-agent"
        "justification": "...",
        "agents": [
            {
                "role": "...",
                "scope": "..."
            },
            ...
        ],
    },
    "should_stop": True, // or False?
    "number_of_iterations": 2 // or anything >1
}
```

## User Input Refiner
***Purpose:*** Improves and standardizes the input so other agents or search tools can interpret it better.

***Input:*** 
```json
{
    "user_input": "..."
}
```
***Output:*** 
```json
{
    "corrected_original": "...",
    "refined_text": "..."
}
```

## Agent
***Purpose:*** 

***Input:*** 
```json

```
***Output:*** 
```json

```