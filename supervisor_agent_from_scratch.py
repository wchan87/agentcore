import logging
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, InjectedToolCallId, BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_ollama import ChatOllama
from langchain_postgres import PGEngine, PGVectorStore
from langchain_tavily import TavilySearch
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command, Send
from typing import Annotated
from supervisor_agent import pretty_print_messages

# START of duplicate code from supervisor_agent.py
POSTGRES_USER: str = 'langchain'
POSTGRES_PASSWORD: str = 'langchain'
POSTGRES_HOST: str = 'localhost'
POSTGRES_PORT: str = '5432'
POSTGRES_DB: str = 'langchain'
TABLE_NAME: str = 'vectorstore'
CONNECTION_STRING: str = f'postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
OLLAMA_MODEL_ID: str = 'llama3.2:3b'
OLLAMA_BASE_URL: str = 'http://localhost:11434'
TAVILY_API_KEY: str = os.environ.get('TAVILY_API_KEY')

pg_engine: PGEngine = PGEngine.from_connection_string(url=CONNECTION_STRING)
embeddings: Embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_ID, base_url=OLLAMA_BASE_URL)
vector_store: VectorStore = PGVectorStore.create_sync(
    engine=pg_engine,
    table_name=TABLE_NAME,
    embedding_service=embeddings
)
ollama_model: BaseChatModel = ChatOllama(model=OLLAMA_MODEL_ID, base_url=OLLAMA_BASE_URL)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)

@tool
def retrieve_docs_from_vector_store(query: Annotated[str, 'The query for similarity search of vector store']) \
    -> Annotated[list[Document], 'List of documents that scored high on similarity search of vector store']:
    """
    Queries a vector store or knowledge base containing consumer reports about vehicles

    :param query: The query for similarity search of vector store
    :return: List of documents that scored high on similarity search of vector store
    """
    logger.info(f"Retrieving documents from vector store: {query}")
    return vector_store.similarity_search(query)

product_insight_agent: CompiledStateGraph = create_react_agent(
    model=ollama_model,
    tools=[retrieve_docs_from_vector_store],
    prompt='''
    You are a product insight agent that can retrieve documents from a vector store given a query
    INSTRUCTIONS:
    - After you're done with your tasks, respond to the supervisor directly
    - Respond ONLY with the results of your work, do NOT include ANY other text.
    - Supervisor CAN NOT handle coding so focus on the business context.
    ''',
    name='product_insight_agent'
)

# for chunk in product_insight_agent.stream({"messages": [{"role": "user", "content": "Technical specifications and general vehicle overview for Tesla Cybertruck"}]}):
#     pretty_print_messages(chunk)

web_search: TavilySearch = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    max_results=3
)
market_analysis_agent: CompiledStateGraph = create_react_agent(
    model=ollama_model,
    tools=[web_search],
    prompt='''
    You are a market analysis agent that can search the web for information relevant to marketing
    INSTRUCTIONS:
    - After you're done with your tasks, respond to the supervisor directly
    - Respond ONLY with the results of your work, do NOT include ANY other text.
    ''',
    name='market_analysis_agent'
)
# END of duplicate code from supervisor_agent.py

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f'transfer_to_{agent_name}'
    description = description or f'Ask {agent_name} for help.'

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            'role': 'tool',
            'content': f'Successfully transferred to {agent_name}',
            'name': name,
            'tool_call_id': tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={**state, 'messages': state['messages'] + [tool_message]},
            graph=Command.PARENT,
        )
    return handoff_tool

# variant of create_handoff_tool to "formulate a task explicitly"
def create_task_description_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f'transfer_to_{agent_name}'
    description = description or f'Ask {agent_name} for help.'

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            'Description of what the next agent should do, including all of the relevant context.',
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {'role': 'user', 'content': task_description}
        agent_input = {**state, 'messages': [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool

# Handoffs
assign_to_product_insight_agent: BaseTool = create_handoff_tool(
    agent_name='product_insight_agent',
    description='Assign task to a product insight agent.',
)
assign_to_market_analysis_agent: BaseTool = create_handoff_tool(
    agent_name='market_analysis_agent',
    description='Assign task to a market analysis agent.',
)
# Variant Handoffs
assign_to_product_insight_agent_with_description: BaseTool = create_task_description_handoff_tool(
    agent_name='product_insight_agent',
    description='Assign task to a product insight agent.',
)
assign_to_market_analysis_agent_with_description: BaseTool = create_task_description_handoff_tool(
    agent_name='market_analysis_agent',
    description='Assign task to a market analysis agent.',
)

supervisor_agent: CompiledStateGraph = create_react_agent(
    model=ollama_model,
    # tools=[assign_to_product_insight_agent, assign_to_market_analysis_agent],
    tools=[assign_to_product_insight_agent_with_description, assign_to_market_analysis_agent_with_description],
    prompt='''
    You are a supervisor agent managing two agents:
    - a product insight agent. Assign research-related tasks about products, specifically automotive vehicles to this agent
    - a market analyst agent. Assign web search tasks about marketing on automotive products and trends to this agent
    Assign work to one agent at a time, do not call agents in parallel.
    Do not do any work yourself.
    ''',
    name='supervisor_agent'
)

supervisor = (
    StateGraph(MessagesState)
        .add_node(supervisor_agent, destinations=('product_insight_agent', 'market_analysis_agent', END))
        .add_node(product_insight_agent)
        .add_node(market_analysis_agent)
        .add_edge(START, 'supervisor_agent')
        .add_edge('product_insight_agent', 'supervisor_agent')
        .add_edge('market_analysis_agent', 'supervisor_agent')
        .compile()
)


for chunk in supervisor.stream({"messages": [{"role": "user", "content": "market size and growth trends for the specific product/industry in the target market for Tesla Cybertruck"}]}):
    pretty_print_messages(chunk, last_message=True)
# final_message_history = chunk["supervisor"]["messages"]
