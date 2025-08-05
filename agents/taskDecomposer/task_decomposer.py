from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph
from langgraph.constants import END, START

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from typing import Literal, List, Optional
from pydantic import BaseModel, Field

import prompts

''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEBUG = os.getenv("DEBUG")

''' Ouput Schemas '''
class SingleAgentPlan(BaseModel):
    role: str = Field(description= 'Role of the agent.')
    scope: str = Field(description= 'Scope of the agent\'s responsibility.')

    def __eq__(self, other: 'SingleAgentPlan') -> bool:
        return self.role == other.role and self.scope == other.scope

class TaskPlan(BaseModel):
    type: Literal['single-agent', 'multi-agent'] = Field(description= 'Whether the task requires a single agent or multiple agents.')
    justification: str = Field(description= 'Justification for the agent\'s type assignment.')
    agents: List[SingleAgentPlan] = Field(description= 'List of agents involved in the task.')

    def __eq__(self, other: 'TaskPlan') -> bool:
        return (
            self.type == other.type and 
            len(self.agents) == len(other.agents) and 
            all(agent == other_agent for agent, other_agent in zip(self.agents, other.agents))
        )

''' Input Schema '''
class TaskDecomposerState(BaseModel):
    user_input: str
    output: Optional[TaskPlan] = None
    error: Optional[str] = None
    should_stop: bool = False
    number_of_iterations: int = 1
    
    def increase(self) -> None:
        self.number_of_iterations += 1

    def stop(self) -> bool:
        print('[TASK_DECOMPOSER_STATE] stop') if DEBUG else None
        self.should_stop = True


''' Tools '''
tavily_tool = TavilySearch(
    tavily_api_key= TAVILY_API_KEY
)


''' LLM '''
llm = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'deepseek/deepseek-chat-v3-0324:free', 
    temperature=0
)


''' Nodes'''
def tavily_search(state: TaskDecomposerState) -> TaskDecomposerState:
    '''
    Use Tavily to search the web for relevant information about the topic.
    '''
    print('\n[NODE] tavily_search') if DEBUG else None
    try:
        # Run the search using the original prompt
        result = tavily_tool.invoke(state.user_input)

        # Parse the output to get a summary of the web findings
        summary_lines = []
        for entry in result.get("results", []):
            title = entry.get("title", "Untitled")
            content = entry.get("content", "")[:500].strip()

            summary_lines.append(f"- {title}\n  {content}\n")

        web_summary = "\n---\n".join(summary_lines) if summary_lines else "No relevant web results found."

        # Format the new input with web findings appended
        enriched_input = (
            f'{state.user_input.strip()}\n\n'
            f'#### Web Findings:\n{web_summary}\n'
        )

        print(f'[NODE] [INFO] Original prompt: {state.user_input}\nEnriched prompt: {enriched_input}') if DEBUG else None

        return state.model_copy(update={'user_input': enriched_input})
    
    except Exception as e:
        print('[NODE] [ERR]', e) if DEBUG else None
        return state

def decompose_prompt(state: TaskDecomposerState) -> TaskDecomposerState:
    '''
    This node decomposes the task into subtasks, assigned to agents.
    '''
    print('\n[NODE] decompose_prompt') if DEBUG else None

    try:
        # format the prompt
        agent_ideas = (
            json.dumps(state.output.model_dump(), indent=4)
            if state.output else "{}"
        )

        prompt = prompts.TASK_DECOMPOSER_PROMPT.format(
            agent_ideas= agent_ideas,
            user_input= state.user_input
        )

        # call the LLM, and ensure it returns a TaskPlan
        response = llm.with_structured_output(TaskPlan).invoke(
            [SystemMessage(content= prompt)]# + [HumanMessage(content= state.user_input)]
        )

        # update the state
        if state.output and state.output == response:
            state.stop()
        else:
            state.increase()

        if DEBUG and state.output:
            print(f'[NODE] [INFO] Previous output: {state.output.model_dump_json()}')
        print(f'[NODE] [INFO] New output: {response.model_dump_json()}') if DEBUG else None

        return state.model_copy(update={'output': response})
    except Exception as e:
        # update the state, if an error occurs
        print('[NODE] [ERR]', e) if DEBUG else None
        return state.model_copy(update={'error': str(e)})


''' Conditional Nodes '''
def should_continue(state: TaskDecomposerState) -> Literal['decompose_prompt', '__end__']:
    '''
    This node checks if the task decomposition should continue.
    '''
    print('\n[CONDITION] should_continue') if DEBUG else None
    return 'end' if state.should_stop else 'retry'


''' Graph '''
task_decomposer_graph = StateGraph(TaskDecomposerState)

task_decomposer_graph.add_node('decompose_prompt', decompose_prompt)
task_decomposer_graph.add_node('tavily_search', tavily_search)

task_decomposer_graph.add_edge(START, 'tavily_search')
task_decomposer_graph.add_edge('tavily_search', 'decompose_prompt')
task_decomposer_graph.add_conditional_edges(
    'decompose_prompt', 
    should_continue,
    {
        'retry': 'decompose_prompt',
        'end': END
    }
)

task_decomposer_app = task_decomposer_graph.compile()


''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(task_decomposer_app.get_graph().draw_mermaid_png())
    if not os.path.exists('./graphs'):
        os.makedirs('./graphs')
    with open(f'./graphs/task_decomposer_app.png', 'wb') as f:
        f.write(task_decomposer_app.get_graph().draw_mermaid_png())


    # Run the graph
    # 'I want an agent that knows about dolphins.'
    user = TaskDecomposerState(user_input= 'I want an agent that thinks and creates tools for LLM agents to use, for specific topics, which will be provided by the user.')
    result = task_decomposer_app.invoke(user)

    print('\n[MAIN]', result)

    print('\n[MAIN]', result['number_of_iterations'])

    print('\n[MAIN]', result['output'].model_dump_json(indent=2))

