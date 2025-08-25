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

from langgraph.graph import StateGraph, MessagesState
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
class InputSchema(MessagesState):
    user_input: str = Field(
        description= 'The user input to be clarified and refined.'
    )

''' Output Schema '''
class OutputSchema(BaseModel):
    corrected_original: str = Field(
        description= 'The original request with grammar and spelling fixed, vocabulary unchanged.'
    )
    refined_text: Optional[str] = Field(
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
).with_structured_output(OutputSchema)



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
def correct_user_input(state: InputSchema) -> InputSchema:
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
            'user_input': corrected,
            'messages': [SystemMessage(content= f'The user\'s request: {corrected}')]
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return {
            'user_input': user_input,
            'messages': [SystemMessage(content= f'This is the user\'s request: {user_input}')]
        }
    
def clarify(state: InputSchema) -> InputSchema:
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
        user_input = input('\n> ')

        return {'messages': [AIMessage(content = clarification.content), HumanMessage(user_input)]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

def refine_user_input(state: InputSchema) -> OutputSchema:
    '''
    This node accepts a corrected version of a user input and provides a refined version of it.
    '''
    print(f'\n{BLUE}[NODE]{RESET} refine_user_input') if DEBUG else None
    try:
        
        history = []
        for mess in state['messages']:
            history.append(mess.pretty_repr() if isinstance(mess, BaseMessage) else str(mess))
            
        prompt = prompts.REFINE_INPUT_PROMPT.format(
            user_input_json= json.dumps(state['user_input']),
            history= '\n---\n\n'.join(history)
        )

        # call the LLM to refine
        refined = refiner.invoke([SystemMessage(content= prompt)])

        print(f'{BLUE}[NODE] [INFO]{RESET} Refined: {refined}') if DEBUG else None

        return refined

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        
        # Return the original user input
        return OutputSchema(corrected_original= state['user_input'], refined_text= None)



''' Conditional Functions '''
def keep_clarifying(state: InputSchema) -> Literal['clarify', 'tools', 'refine']:
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
            input('\n> ')

        return 'tools'

    # Otherwise, keep asking for clarifications
    print(f'{BLUE}[NODE] [INFO]{RESET} Will ask for clarifications') if DEBUG else None
    return 'clarify'

def refinement_okay(state: OutputSchema) -> Literal['__end__', 'refine']:
    # TODO: can loop back to clarify
    '''
    This node asks the user if the refined version of the user input is okay.
    '''
    print(f'\n{BLUE}[NODE]{RESET} refinement_okay') if DEBUG else None
    print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {state.refined_text}')
    answer = input(f'{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay? (y/n) > ')

    if answer.lower() in ['y', 'ye', 'yea', 'yes']:
        return END
    else:
        state.add_user_request(input('{GREEN}[NODE] [CONFIRMATION]{RESET} Please insert your request >'))
        return 'refine'

''' Graph '''
input_refiner_graph = StateGraph(InputSchema, output_schema= OutputSchema)

input_refiner_graph.add_node('correct', correct_user_input)
input_refiner_graph.add_node('clarify', clarify)
input_refiner_graph.add_node('tools', ToolNode([tavily_search]))
input_refiner_graph.add_node('refine', refine_user_input)

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
input_refiner_graph.add_edge('refine', END)
# input_refiner_graph.add_conditional_edges(
#     'refine',
#     lambda state: 'refine_tools' if _will_tool_call(state['messages']) else END,
#     {   # Not needed, but added for clarity
#         'refine_tools': 'refine_tools',
#         '__end__': END
#     }
# )

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

    user = {'user_input': 'i want an agent to help me with desinging a holiday.'}
    response = input_refiner_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')