import asyncio
import boto3
import json

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore
from langchain_ollama import OllamaEmbeddings

POSTGRES_USER: str = 'langchain'
POSTGRES_PASSWORD: str = 'langchain'
POSTGRES_HOST: str = 'localhost'
POSTGRES_PORT: str = '5432'
POSTGRES_DB: str = 'langchain'
TABLE_NAME: str = 'vectorstore'
VECTOR_SIZE: int = 3072
S3_BUCKET: str = 'raw'
S3_ACCESS_KEY: str = 'admin'
S3_SECRET_KEY: str = 'password'
MINIO_ENDPOINT: str = 'http://localhost:9000'
DOCUMENTS: list[str] = [ 'output-1.json', 'output-2.json', 'output-3.json' ]
OLLAMA_MODEL: str = 'llama3.2:3b'
OLLAMA_BASE_URL: str = 'http://localhost:11434'

CONNECTION_STRING: str = (
    f'postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}'
    f':{POSTGRES_PORT}/{POSTGRES_DB}'
)
pg_engine: PGEngine = PGEngine.from_connection_string(url=CONNECTION_STRING)
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=MINIO_ENDPOINT
)

# 1. Setup the vector store
# TODO consider setup outside of the document loader
async def setup_vector_store_sync():
    await pg_engine.ainit_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=VECTOR_SIZE,
        overwrite_existing=True # TODO this drops the table, may need the equivalent of CREATE TABLE ... IF NOT EXISTS
    )
# TODO is this sync, if not, then should it be setup to wait for the setup just in case of timing issues?
asyncio.run(setup_vector_store_sync())

# 2. Load the JSONs into list[Document]
docs: list[Document] = []
for i, document in enumerate(DOCUMENTS):
    # S3FileLoader runs into "ImportError: unstructured package not found, please install it with `pip install unstructured`"
    # https://github.com/langchain-ai/langchain/issues/7944
    # loader: BaseLoader = S3FileLoader(
    #     S3_BUCKET, document, aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY, endpoint_url=MINIO_ENDPOINT
    # )
    # docs.extend(loader.load())
    # JSONLoader doesn't work either because it only supports file_path being passed to it
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=document)
    content: str = obj['Body'].read().decode('utf-8')
    # partially derived from https://github.com/langchain-ai/langchain-community/blob/90860265dd6f0a9e840b8350ba8e8b2502225d51/libs/community/langchain_community/document_loaders/json_loader.py#L153-L168
    docs.append(Document(page_content=content, metadata={ 'source': f's3://{S3_BUCKET}/{document}', 'document_seq_num': i + 1 }))

# 3. Recursively split the JSON into manageable chunks
all_split_docs: list[Document] = []
json_splitter = RecursiveJsonSplitter(max_chunk_size=1000)
for document in docs:
    # note that the output-#.json files are actually JSON arrays
    split_docs: list[Document] = json_splitter.create_documents(texts=json.loads(document.page_content))
    # original metadata is lost with this approach so this is rehydrating it
    # TODO confirm if there's a better way to do this
    for i, split_doc in enumerate(split_docs):
        split_doc.metadata = {
            'source': document.metadata['source'],
            'document_seq_num': document.metadata['document_seq_num'],
            'document_chunk_seq_num': i + 1
        }
    all_split_docs.extend(split_docs)

# 4. Setup PGVectorStore to load it with documents
async def get_vector_store_async() -> VectorStore:
    embeddings: Embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_NAME,
        embedding_service=embeddings,
    )
vector_store: VectorStore = asyncio.run(get_vector_store_async())
# TODO it takes a 10-20 min when running locally so consider chunking the number of documents
vector_store.add_documents(all_split_docs)
