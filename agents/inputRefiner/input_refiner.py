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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, RemoveMessage, ToolMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.constants import END, START
from langgraph.checkpoint.memory import MemorySaver

import os
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv

from typing import TypedDict, Optional, Annotated, List, Literal
from pydantic import BaseModel, Field

from agents.inputRefiner import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'

print(f'{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Input Refiner') if DEBUG else None



""" Schemas """
''' Input Schema '''
class InputSchema(TypedDict):
    user_input: str = Field(
        description= 'The user input to be clarified and refined.'
    )

''' Intermediate Schema '''
class IntermediateSchema(MessagesState):
    corrected_original: str = Field(
        description= 'The original request with grammar and spelling fixed, vocabulary unchanged.'
    )
    refinements: Annotated[List[AIMessage], add_messages] = Field(
        description= 'The LLM refinements, if any.'
    )
    user_requests: List[HumanMessage] = Field(
        description= 'The user requests, if any.'
    )

''' Output Schema '''
class OutputSchema(BaseModel):
    corrected_original: str = Field(
        description= 'The original request with grammar and spelling fixed, vocabulary unchanged.'
    )
    refined_text: str = Field(
        description= 'A more precise, clear, and search-friendly version of the request.'
    )



''' Tools '''
tavily_search = TavilySearch(
    tavily_api_key= TAVILY_API_KEY,
    search_depth= "advanced",
    max_results= 5,
    include_answer= True
).as_tool()



''' LLM '''
correcter =  ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1',
    api_key= OPENROUTER_API_KEY,
    model= 'moonshotai/kimi-k2:free',
    temperature= 0
)

clarifier = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1',
    api_key= OPENROUTER_API_KEY,
    model= 'moonshotai/kimi-k2:free',
    temperature= 0.8
).bind_tools([tavily_search])

refiner = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'moonshotai/kimi-k2:free', 
    temperature= 0.7
)



''' Helpful Functions '''
def _will_tool_call(messages: list[BaseMessage], actually_called: bool= False) -> bool:
    '''
    Check if the last message will call a tool.

    ### Args:
    - `messages`: the list of messages up to now
        - **note:** remember to add the last message if the state is not updated yet
    - `actually_call`: whether it actually called the tool
        - **default**: False

    ### Returns:
    - True if the last message will call a tool

    ### Tool Calls:
    - 'will use tavily web search to gather context'
        - Skipped if actually_call is True
    - last_message.tool_calls exists and not empty
    - last_message.additional_kwargs.tool_calls exists and not empty
    - tools_condition(last_message) == tools
    '''
    last_message = messages[-1]
    return (
        # If actually_called is set, we should check wheather the last message is a tool call, not the content
        'will use tavily web search to gather context' in last_message.content.lower() and not actually_called or 
        hasattr(last_message, 'tool_calls') and last_message.tool_calls or
        hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls', False) or
        tools_condition(messages) == 'tools'
    )



''' Nodes'''
def correct_user_input(state: InputSchema) -> IntermediateSchema:
    '''
    This node accepts a user input and provides a corrected version of it.
    '''
    print(f'\n{BLUE}[NODE]{RESET} correct_user_input') if DEBUG else None
    try:
        # prompt
        user_input = state['user_input']
        prompt = prompts.CORRECTION_PROMPT.format(user_input= user_input)
        # call the LLM
        corrected = correcter.invoke(prompt).content

        print(f'{BLUE}[NODE] [INFO] [CORRECTION]{RESET} {corrected}') if DEBUG else None

        # Replace the message with the corrected one
        return {
            'messages': [HumanMessage(content= corrected)],
            'corrected_original': corrected,
            'refinements': [],
            'user_requests': []
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return {
            'messages': [HumanMessage(content= user_input)],
            'corrected_original': user_input,
            'refinements': [],
            'user_requests': []
        }
    
def clarify(state: IntermediateSchema) -> IntermediateSchema:
    '''
    This node accepts a corrected version of a user input and provides a refined version of it.
    '''
    print(f'\n{BLUE}[NODE]{RESET} clarify') if DEBUG else None
    if DEBUG and isinstance(state['messages'][-1], ToolMessage):
        print(f'{GREEN}[NODE] [TAVILY RESULT]{RESET} {state["messages"][-1].content}')
    try:
        # prompt
        prompt = prompts.CLARIFICATION_PROMPT.format(
            user_input= state['messages'][0].content,
            clarifications= '\n---\n'.join([mess.content for mess in state['messages'][1:]])
        )
        # call the LLM
        clarification = clarifier.invoke([SystemMessage(content= prompt)])
        print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {clarification}') if DEBUG else None

        if 'no clarification needed' in clarification.content.lower():
            print(f'{BLUE}[NODE] [INFO]{RESET} No further clarifications needed') if DEBUG else None
            return {'messages': [AIMessage(content = clarification.content)]}
        
        if _will_tool_call(state['messages'] + [clarification]):
            print(f'{BLUE}[NODE] [INFO]{RESET} Will use tavily web search to gather context') if DEBUG else None
            return {'messages': [clarification]}

        print(f'{GREEN}[NODE] [CLARIFICATION/ASSUMPTION QUESTION]{RESET} {clarification.content}')
        user_input = input(f'\n{GREEN}[NODE] [INPUT] >{RESET} ')

        return {'messages': [AIMessage(content = clarification.content), HumanMessage(user_input)]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

def refine_user_input(state: IntermediateSchema) -> IntermediateSchema:
    '''
    This node accepts a corrected version of a user input and provides a refined version of it.
    '''
    print(f'\n{BLUE}[NODE]{RESET} refine_user_input') if DEBUG else None
    try:
        
        history: list[str] = []
        for mess in state['messages']:
            history.append(mess.pretty_repr() if isinstance(mess, BaseMessage) else str(mess))

        refinements_and_requests: list[str] = []
        refinements = state['refinements']
        requests = state['user_requests']
        for i in range(max(len(refinements), len(requests))):
            if i < len(refinements):
                refinements_and_requests.append(refinements[i].pretty_repr())
            if i < len(requests):
                refinements_and_requests.append(requests[i].pretty_repr())
            
        prompt = prompts.REFINE_INPUT_PROMPT.format(
            history= '\n---\n\n'.join(history),
            refinements_and_requests= '\n---\n\n'.join(refinements_and_requests)
        )

        # call the LLM to refine
        refined = refiner.invoke([SystemMessage(content= prompt)]).content

        print(f'{BLUE}[NODE] [INFO]{RESET} Refined: {refined}') if DEBUG else None

        return {'refinements': [AIMessage(content= refined)]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        
        # Return the original
        return state

def parse_output(state: IntermediateSchema) -> OutputSchema:
    '''
    This node accepts a corrected version of a user input and provides a refined version of it.
    '''
    print(f'\n{BLUE}[NODE]{RESET} parse_output') if DEBUG else None
    return OutputSchema(corrected_original= state['corrected_original'], refined_text= state['refinements'][-1].content)



''' Conditional Functions '''
def keep_clarifying(state: IntermediateSchema) -> Literal['clarify', 'tools', 'refine']:
    print(f'\n{BLUE}[NODE]{RESET} keep_clarifying') if DEBUG else None

    # If no further clarifications are needed
    if 'no clarification needed' in state['messages'][-1].content.lower():
        print(f'{BLUE}[NODE] [INFO]{RESET} No further clarifications needed') if DEBUG else None
        return 'refine'
    
    # If a tool call is needed
    if isinstance(state['messages'][-1], AIMessage) and _will_tool_call(state['messages']):
        print(f'{BLUE}[NODE] [INFO]{RESET} Will use tavily web search to gather context') if DEBUG else None
        # But no actually tool call happened
        while not _will_tool_call(state['messages'], actually_called= True): # TODO:
            sys_msg = prompts.FORCE_TOOL_CALL
            # Call the llm again to make it call the tool
            state['messages'] += [clarifier.invoke([state['messages'][-1], SystemMessage(content= sys_msg)])]
            print(f'{BLUE}[NODE] [INFO]{RESET} Trying to call the tool.') if DEBUG else None
            input('\n> press to continue') if DEBUG else None

        return 'tools'

    # Otherwise, keep asking for clarifications
    print(f'{BLUE}[NODE] [INFO]{RESET} Will ask for clarifications') if DEBUG else None
    return 'clarify'

def refinement_okay(state: IntermediateSchema) -> Literal['parse_output', 'refine']:
    '''
    This node asks the user if the refined version of the user input is okay.
    '''
    print(f'\n{BLUE}[NODE]{RESET} refinement_okay') if DEBUG else None

    # Ask the user if the refined version of the user input is okay
    print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {state["refinements"][-1].content}')
    answer = input(f'{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ')

    # If the answer is yes, parse the output and end
    if answer.lower() in ['y', 'ye', 'yea', 'yes', 'ok', 'okay', 'k']:
        return 'parse_output'

    # If the answer is no, ask for a new request, and keep refining
    else:
        state['user_requests'] += [HumanMessage(answer)]
        return 'refine'



''' Graph '''
input_refiner_graph = StateGraph(IntermediateSchema, input_schema= InputSchema, output_schema= OutputSchema)

input_refiner_graph.add_node('correct', correct_user_input)
input_refiner_graph.add_node('clarify', clarify)
input_refiner_graph.add_node('tools', ToolNode([tavily_search]))
input_refiner_graph.add_node('refine', refine_user_input)
input_refiner_graph.add_node('parse_output', parse_output)

input_refiner_graph.add_edge(START, 'correct')
input_refiner_graph.add_edge('correct', 'clarify')
input_refiner_graph.add_conditional_edges(
    'clarify', 
    keep_clarifying,
    {   # Not needed, but added for clarity
        'clarify': 'clarify',
        'tools': 'tools',
        'refine': 'refine'
    }
)
input_refiner_graph.add_edge('tools', 'clarify')
input_refiner_graph.add_conditional_edges(
    'refine',
    refinement_okay,
    {   # Not needed, but added for clarity
        'parse_output': 'parse_output',
        'refine': 'refine'
    }
)
input_refiner_graph.add_edge('parse_output', END)

input_refiner_app = input_refiner_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(input_refiner_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/input_refiner_app.png', 'wb') as f:
        f.write(input_refiner_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'inputRefiner'
    os.environ['LANGSMITH_PROJECT'] = 'inputRefiner'
    client = Client()

    config = {
        'configurable': {
            'user_id': 'inputRefiner',
            'run_name': 'inputRefiner',
            'thread_id': 'inputRefiner'
        }
    }

    user = {'user_input': 'make assumptions only-no clarifications will be answered. i want an agent to help me with automating online orders. The order are of food.'}
    response = input_refiner_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')