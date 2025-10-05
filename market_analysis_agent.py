import boto3
import logging
import os
import uuid
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from tavily import TavilyClient
from typing import Any, TypedDict

OLLAMA_MODEL: str = 'llama3.2:3b'
OLLAMA_BASE_URL: str = 'http://localhost:11434'
TAVILY_API_KEY: str = os.environ.get('TAVILY_API_KEY')
S3_BUCKET: str = 'analysis'
S3_ACCESS_KEY: str = 'admin'
S3_SECRET_KEY: str = 'password'
MINIO_ENDPOINT: str = 'http://localhost:9000'

client: TavilyClient = TavilyClient(TAVILY_API_KEY)
llm: BaseChatModel = init_chat_model(OLLAMA_MODEL, model_provider='ollama', base_url=OLLAMA_BASE_URL)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=MINIO_ENDPOINT
)

class State(TypedDict):
    question: str
    product: str
    api_responses: list[dict[str, Any]]
    answer: str
    s3_result_location: str

def extract(state: State) -> dict[str, Any]:
    question: str = state['question']
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
        '''
        # Role: Market Analyst for AwsomeCar
        
        # Goal: Extract the core product/topic from the question.
        
        # Guideline:
        1. Extract the main product from the questions like "Tesla Model Y에 대한 마케팅 보고서 작성해줘" → extract "Tesla Model Y"
        2. You MUST ONLY return the product or topic without any additional text
        3. You MUST NOT PROVIDE code for performing this extraction
        
        # Product Extraction Rules:
        - Look for vehicle names, models, or automotive products in the user's request
        - Ignore instructions like "Write a marketing report", "Analyze this", etc.
        - If multiple products are mentioned, focus on the main one - Examples:
            - "Write a marketing report on Tesla Model Y" → "Tesla Model Y"
            - "Compare and analyze Hyundai IONIQ 5 and Kia EV6" → "Hyundai IONIQ 5, Kia EV6"
            - "Tell me about the latest electric vehicle market trends" → "Electric vehicles"
        
        Question: {question}
        '''
    )
    messages: PromptValue = prompt.invoke({'question': state['question']})
    ai_message: BaseMessage = llm.invoke(messages)
    product: str = ai_message.content
    logger.info(f'Extracted product: "{product}" from question: "{question}"')
    return {'product': product}

def search_tavily(state: State) -> dict[str, Any]:
    product: str = state['product']
    search_queries: list[str] = [
        'market size and growth trends for the specific product/industry in the target market',
        'detailed competitor analysis including market share and strategies',
        'target audience demographics, psychographics, and behaviors in the specific market',
        'regulatory factors, local market challenges, and consumer preferences'
    ]
    api_responses: list[dict[str, Any]] = []
    for search_query in search_queries:
        query: str = f'Provide {search_query} regarding the {product}'
        logger.info(f'Query: "{query}" sent to Tavily')
        api_responses.append(client.search(query=query))
    return {'api_responses': api_responses}

def generate(state: State) -> dict[str, Any]:
    # TODO the output format isn't adhered to exactly so need to enforce it somehow...
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
        '''
        # Role: Market Analyst for AwsomeCar

        # Goal: Provide comprehensive market research based on search engine results pages and format the results in a standardized format that can be stored for further processing.
        Product or Trend: {product}
        
        # Guideline:
        1. CRITICAL: For EACH data point, statistic, or claim in your analysis, you MUST include the reference source

        # Standardized Report Format:
        After completing your research, you MUST format your findings in the following standardized report format:
        ```
        ==================================================
        ## Analysis Stage: [Automotive Products and Trends Market Research]
        ## REFERENCE: [Main Reference Source]
        ## Execution Time: [Current Date and Time]
        --------------------------------------------------
        Result Description:

        [Detailed market analysis in bullet points or paragraphs]

        Key points:
        1. [First key insight]
        2. [Second key insight]
        3. [Third key insight]
        4. [Fourth key insight]
        --------------------------------------------------
        ==================================================
        ```

        CRITICAL INSTRUCTION:
        1. NEVER ask clarifying questions to the user
        2. ENSURE all sections contain DETAILED content with specific examples and data points
        3. CRITICAL: Format your analysis according to the template specified

        Search Engine Result Pages: <search_results>{api_responses}</search_results>
        '''
    )
    product: str = state['product']
    messages: PromptValue = prompt.invoke({'product': product, 'api_responses': state['api_responses']})
    logger.info(f'LLM invoked for market analysis on product or trend: "{product}"')
    ai_message: BaseMessage = llm.invoke(messages)
    return {'answer': ai_message.content}

def store_results_in_s3(state: State) -> dict[str, Any]:
    key: str = f'market_analysis_agent/{uuid.uuid4()}.txt'
    s3_location: str = f's3://{S3_BUCKET}/{key}'
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=state['answer'], ContentType='text/plain')
    logger.info(f'Market analysis saved to {s3_location}')
    return {'s3_result_location': s3_location}

graph_builder: StateGraph = StateGraph(State)
graph_builder.add_sequence([extract, search_tavily, generate, store_results_in_s3])
graph_builder.add_edge(START, 'extract')
graph_builder.add_edge('store_results_in_s3', END)
graph: CompiledStateGraph = graph_builder.compile()
logger.info(graph.get_graph().draw_ascii())

response: dict[str, Any] = graph.invoke({'question': 'Tell me about Tesla Cybertruck and summarize consumer reactions.'})
print(response['s3_result_location'])
