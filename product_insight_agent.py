import asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from typing import Any
from typing_extensions import List, TypedDict

POSTGRES_USER: str = 'langchain'
POSTGRES_PASSWORD: str = 'langchain'
POSTGRES_HOST: str = 'localhost'
POSTGRES_PORT: str = '5432'
POSTGRES_DB: str = 'langchain'
TABLE_NAME: str = 'vectorstore'
OLLAMA_MODEL: str = 'llama3.2:3b'
OLLAMA_BASE_URL: str = 'http://localhost:11434'
CONNECTION_STRING: str = (
    f'postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}'
    f':{POSTGRES_PORT}/{POSTGRES_DB}'
)
pg_engine: PGEngine = PGEngine.from_connection_string(url=CONNECTION_STRING)
async def get_vectorstore_async() -> VectorStore:
    embeddings: Embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_NAME,
        embedding_service=embeddings,
    )
vector_store: VectorStore = asyncio.run(get_vectorstore_async())
llm = init_chat_model(OLLAMA_MODEL, model_provider='ollama', base_url=OLLAMA_BASE_URL)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State) -> dict[str, Any]:
    retrieved_docs: List[Document] = vector_store.similarity_search(state['question'])
    return {'context': retrieved_docs}

def generate(state: State) -> dict[str, Any]:
    # this code snippet pulls from https://smith.langchain.com/hub/rlm/rag-prompt
    # from langchain import hub
    # prompt: ChatPromptTemplate = hub.pull('rlm/rag-prompt')
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
        '''
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        '''
    )
    docs_content = '\n\n'.join(doc.page_content for doc in state['context'])
    messages: PromptValue = prompt.invoke({'question': state['question'], 'context': docs_content})
    ai_message: BaseMessage = llm.invoke(messages)
    return {'answer': ai_message.content}

# Compile application and test
graph_builder: StateGraph = StateGraph(State)
graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph: CompiledStateGraph = graph_builder.compile()

response: dict[str, Any] = graph.invoke({'question': 'Summarize the product information and customer feedback for Tesla Model Y in English.'})
print(response['answer'])
