import logging
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import convert_to_messages
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGEngine, PGVectorStore
from langchain_tavily import TavilySearch
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain_ollama.chat_models import ChatOllama
from langgraph_supervisor import create_supervisor
from typing import Annotated

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

# START of code snippet for debugging copied from https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        # this change is needed with Step 4 due to node_update being None after transfer back to supervisor
        if node_update:
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                pretty_print_message(m, indent=is_subgraph)
        else:
            print("node_update was set to None due to delegation task")
        print("\n")
# END of code snippet for debugging

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

# for chunk in product_insight_agent.stream({"messages": [{"role": "user", "content": "market size and growth trends for the specific product/industry in the target market for Tesla Cybertruck"}]}):
#     pretty_print_messages(chunk)

supervisor: CompiledStateGraph = create_supervisor(
    model=ollama_model,
    agents=[product_insight_agent, market_analysis_agent],
    prompt='''
    You are a supervisor agent managing two agents:
    - a product insight agent. Assign research-related tasks about products, specifically automotive vehicles to this agent
    - a market analyst agent. Assign web search tasks about marketing on automotive products and trends to this agent
    Assign work to one agent at a time, do not call agents in parallel.
    Do not do any work yourself.
    ''',
    add_handoff_back_messages=True,
    output_mode='full_history',
).compile()

for chunk in supervisor.stream({"messages": [{"role": "user", "content": "market size and growth trends for the specific product/industry in the target market for Tesla Cybertruck"}]}):
    pretty_print_messages(chunk, last_message=True)
# final_message_history = chunk["supervisor"]["messages"]
