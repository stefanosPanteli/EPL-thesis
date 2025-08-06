"""
- `author:` Stefanos Panteli
- `date:` 2025-08-06
- `description:` Accepts a user input and provides a corrected and refined version of it.

## How to use
1. Import the app. (`from agents.userInputRefiner.user_input_refiner import user_input_refiner_app`)
2. Input a dict with the following keys:
    - `user_input: str`: The user input to be refined.
3. Invoke the app.
4. Get the output dict with the following keys:
    - `corrected_original: str`: The original request with grammar and spelling fixed, vocabulary unchanged.
    - `refined_text: str`: A more precise, clear, and search-friendly version of the request.

## Usage
```python
from agents.userInputRefiner.user_input_refiner import user_input_refiner_app
graph_input = {'user_input': 'I want a personall fitness coach.'}

refined = user_input_refiner_app.invoke(graph_input)

# refined = {
#     'corrected_original': 'I want a personal fitness coach.', 
#     'refined_text': 'I am looking to hire a personal fitness trainer for customized workout guidance and coaching.'
# }
```
"""



''' Imports '''
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph
from langgraph.constants import END, START

import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

from typing import Optional
from pydantic import BaseModel, Field

from agents.userInputRefiner import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DEBUG = os.getenv('DEBUG')

print('\n[AGENT] [INFO] [STARTUP] User Input Refiner') if DEBUG else None



""" Schemas """
''' Input Schema '''
class InputSchema(BaseModel):
    user_input: str

''' Output Schema '''
class OutputSchema(BaseModel):
    corrected_original: str = Field(
        description="The original request with grammar and spelling fixed, vocabulary unchanged."
    )
    refined_text: Optional[str] = Field(
        description="A more precise, clear, and search-friendly version of the request."
    )


''' Tools '''
tavily_tool = TavilySearch(tavily_api_key= TAVILY_API_KEY).as_tool(
    description= 'Use Tavily to search the web for relevant information to help you refine the user input.'
)



''' LLM '''
refiner = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'deepseek/deepseek-chat-v3-0324:free', 
    temperature=0
).bind_tools([tavily_tool]).with_structured_output(OutputSchema)



''' Nodes'''
def refine_user_input(state: InputSchema) -> OutputSchema:
    print('\n[NODE] refine_user_input') if DEBUG else None
    try:
        prompt = prompts.REFINE_INPUT_PROMPT

        # call the LLM, and ensure it returns a TaskPlan
        refined = refiner.invoke(
            [SystemMessage(content= prompt), HumanMessage(content= state.user_input)]
        )

        print(f'[NODE] [INFO] Refined: {refined}') if DEBUG else None

        return refined

    except Exception as e:
        print('[NODE] [ERR]', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return OutputSchema(corrected_original= state.user_input, refined_text= None)



''' Graph '''
user_input_refiner_graph = StateGraph(InputSchema, output_schema= OutputSchema)

user_input_refiner_graph.add_node("refine_user_input", refine_user_input)

user_input_refiner_graph.add_edge(START, "refine_user_input")
user_input_refiner_graph.add_edge("refine_user_input", END)

user_input_refiner_app = user_input_refiner_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(user_input_refiner_app.get_graph().draw_mermaid_png())
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/user_input_refiner_app.png', 'wb') as f:
        f.write(user_input_refiner_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'UserInputRefiner'
    os.environ['LANGSMITH_PROJECT'] = 'UserInputRefiner'
    client = Client()

    config = {
        'configurable': {
            'user_id': 'Test-UserInputRefiner',
            'run_name': 'Test-UserInputRefiner',
            'tags': ['UserInputRefiner', 'deepseek']
        }
    }

    user = InputSchema(user_input= 'I want a personall fitness coach.')
    user_input_refiner_app.invoke(user, config= config)