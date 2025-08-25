# EPL-thesis
Multi-agent system for the thesis.

`author:` Stefanos Panteli

# Run Steps
## To use:
1. First make a virtual envirnment.
2. Then install the requirements.
3. Make the API keys, like the `.env.example` file.
    Where to find the keys:
    1. Open Router:
        - To create an account, sign up [https://openrouter.ai/](https://openrouter.ai/).
        - To create an API key, go to [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).
    2. Tavily:
        - To create an account, sign up [https://tavily.com/](https://tavily.com/).
        - To create an API key, go to [https://app.tavily.com/home](https://app.tavily.com/home).
    3. LangSmith/Chain:
        - To create an account, sign up [https://smith.langchain.com/](https://smith.langchain.com/).
        - To create an API key, log in -> Settings -> API Keys
```bash
cd .\to the clone directory

python -m venv venv # Creates the virtual environment named venv
pip install -r requirements.txt # Installs the requirements

#! Modify the .env.example file
cp .env.example .env # Copy the .env.example file to .env
```

## To run:
1. Run the following commands each time you want to start the system.
(From the Clone directory)
```bash
cd .\to the clone directory
.\venv\Scripts\Activate.ps1 # Activate the venv

$env:PYTHONPATH = (Get-Location) # Set the PYTHONPATH (For windows)
#! OR
export PYTHONPATH=$PWD           # Set the PYTHONPATH (For linux)


cd /agents/<agent_name> # Go to the agent folder
python <agent_name>.py # Run the agent
```

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