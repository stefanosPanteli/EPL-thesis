import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the correct number of arguments are provided
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <agent_name>")
    print('    <agent_name> must be in snake_case.')
    sys.exit(1)

agent_name = sys.argv[1]

parent_path = Path(__file__).resolve().parent

# Convert agent_name to a suitable directory name
directory_name = agent_name.split('_')[0] + ''.join(x.title() for x in agent_name.split('_')[1:])

# Create the directory and a 'graphs' subdirectory
os.makedirs(parent_path / 'agents' / directory_name / 'graphs', exist_ok=True)

# List of files to create
FILE_NAMES = ['prompts.py', f'{agent_name}.py']

agent_file_text = f"""\"\"\"
- `author:` Stefanos Panteli
- `date:` {datetime.today().strftime('%Y-%m-%d')}
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.userInputRefiner.{agent_name} import {agent_name}_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.userInputRefiner.{agent_name} import {agent_name}_app
graph_input = {{ # TODO: add }}

refined = {agent_name}_app.invoke(graph_input)

# refined = {{ # TODO: add }}
```
\"\"\"



''' Imports '''
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph
from langgraph.constants import END, START

import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

from typing import Literal
from pydantic import BaseModel, Field

from agents.{directory_name} import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'


print(f'\\n{{BLUE}}[AGENT] [INFO] [STARTUP]{{RESET}} {' '.join(x.title() for x in agent_name.split('_'))}') if DEBUG else None



\"\"\" Schemas \"\"\"
''' General Schemas '''

''' Input Schema '''

''' Intermediate Schemas '''

''' Output Schema '''



''' Tools '''



''' LLM '''
llm = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'deepseek/deepseek-chat-v3-0324:free', 
    temperature= 0
)



''' Helpful Functions '''



''' Nodes'''



''' Conditional Functions '''



''' Graph '''
{agent_name}_graph = StateGraph() # TODO: change


{agent_name}_app = {agent_name}_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image({agent_name}_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/{agent_name}_app.png', 'wb') as f:
        f.write({agent_name}_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = '{directory_name}'
    os.environ['LANGSMITH_PROJECT'] = '{directory_name}'
    client = Client()

    config = {{
        'configurable': {{
            'user_id': '{directory_name}',
            'run_name': '{directory_name}',
            'tags': ['{directory_name}', 'deepseek'] # TODO: add
        }}
    }}

    user = ''' # TODO: add
    response = {agent_name}_app.invoke(user, config= config)

    import json
    print(f'{{BLUE}}[MAIN] [INFO]{{RESET}}', json.dumps(response, indent= 4)) if DEBUG else None
"""
# Create each file in the new directory
for file in FILE_NAMES:
    with open(parent_path / 'agents' / directory_name / file, 'w') as f:
        if file == f'{agent_name}.py':
            f.write(agent_file_text)
