# Amazon Bedrock Multi-Agent Collaboration

Based on [Amazon Bedrock multi-agent collaboration: Building JARVIS (Smart Assistant) to solve complex problems with simple AI Agents](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US)

> Amazon Bedrock Agents is a feature provided by Amazon Bedrock that enables you to build multi-step task automation by seamlessly connecting generative AI applications with your company systems, APIs, and data sources. Amazon Bedrock Agents use foundation model (FM) reasoning, APIs, and data to classify user requests, gather relevant information, and efficiently complete tasks, allowing teams to focus on high-value work. Building agents is simple and fast, requiring just a few steps to complete setup.

This feature seems to be the equivalent to the following concepts in these frameworks:
* LangChain / LangGraph
  * [PyPI > langgraph-supervisor](https://pypi.org/project/langgraph-supervisor/)
    * [GitHub > langchain-ai/langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py)
  * [LangChain Blog > LangGraph: Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
    * [LangGraph > Examples > Agent Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
* Strands Agents
  * [Strands Agents > User Guide > Multi-agent > Agents as Tools](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/agents-as-tools/)
  * [Strands Agents > User Guide > Multi-agent > Multi-agent Patterns](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/multi-agent-patterns)

## Docker Compose Cluster

1. Setup Docker Compose Cluster
    ```bash
    docker compose up -d
    ```
   * Postgres (based on [this](https://python.langchain.com/docs/integrations/vectorstores/pgvectorstore/#setup)) - `localhost:5432` (or `jdbc:postgresql://localhost:5432/langchain`) - Credentials: `langchain` / `langchain`
   * MinIO - http://localhost:9000 - Credentials: `admin` / `password`
   * Ollama - http://localhost:11434
2. Stop Docker Compose Cluster
    ```bash
    docker compose stop
    ```
3. Teardown Docker Compose Cluster
    ```bash
    docker compose down --volume
    ```

### Ollama

The embedding length determines the `VECTOR_SIZE` that is used to store the embeddings in [Postgres](#postgres) via `pgvector`. You can find the embedding length from the model information such as `llama.embedding_length` in [llama3.2:3b](https://ollama.com/library/llama3.2:3b/blobs/dde5aa3fc5ff). Postgres has fixed vector size so if the embedding length is too large, the data load would fail.

### Postgres

If an error is returned about 'Type "vector" does not exist on postgresql', then you may need to run the following on the `langchain` database as per [StackOverflow](https://stackoverflow.com/a/76221780)
```sql
CREATE EXTENSION vector IF NOT EXISTS
```

### MinIO

Copy the documents for [Lab 1](#lab-1---product-insight-agent) to MinIO
```bash
aws --endpoint-url=http://localhost:9000 s3 cp temp/output-1.json s3://raw/
aws --endpoint-url=http://localhost:9000 s3 cp temp/output-2.json s3://raw/
aws --endpoint-url=http://localhost:9000 s3 cp temp/output-3.json s3://raw/
```

## Lab 1 - Product Insight Agent

[Lab 1: Product Insight Agent](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab1-single-agent) involves an agent that searches [Amazon Bedrock Knowledge Base](https://docs.aws.amazon.com/en_us/bedrock/latest/userguide/kb-how-data.html) (i.e., [Retrieval-Augmented Generation (RAG)](https://python.langchain.com/docs/tutorials/rag/)) that's populated via [documents uploaded](https://docs.aws.amazon.com/en_us/bedrock/latest/userguide/kb-data-source-customize-ingestion.html) to S3 bucket which is [chunked](https://docs.aws.amazon.com/en_us/bedrock/latest/userguide/kb-data-source-customize-ingestion.html) and converted to embeddings with the steps as follows:
* [Step 1: Creating S3 Bucket and Uploading Data](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab1-single-agent/step-01-s3-upload)
  * Sample documents to be converted to embeddings
    * [output-1.json](https://static.us-east-1.prod.workshops.aws/public/dbd230e3-b8b3-49a3-9b0e-c717778f99e9/static/lab_product_insight_agent/data/output-1.json)
    * [output-2.json](https://static.us-east-1.prod.workshops.aws/public/dbd230e3-b8b3-49a3-9b0e-c717778f99e9/static/lab_product_insight_agent/data/output-2.json)
    * [output-3.json](https://static.us-east-1.prod.workshops.aws/public/dbd230e3-b8b3-49a3-9b0e-c717778f99e9/static/lab_product_insight_agent/data/output-3.json)
* [Step 2: Creating Amazon Bedrock Knowledge Base](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab1-single-agent/step-02-kb-create)
* [Step 3: Create Product Insight Agent](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab1-single-agent/step-03-agent-config)

The local equivalent of the above architecture is
* use LangChain to leverage [Documents and Document Loaders](https://python.langchain.com/docs/tutorials/retrievers/#documents-and-document-loaders) to generate embeddings from documents to be uploaded to MinIO
* use LangChain [PGVectorStore](https://python.langchain.com/docs/integrations/vectorstores/pgvectorstore/) to store the embeddings (i.e., Postgres with [pgvector](https://github.com/pgvector/pgvector) extension is already setup on DockerHub [pgvector/pgvector](https://hub.docker.com/r/pgvector/pgvector))
* use LangChain [Retriever](https://python.langchain.com/docs/tutorials/retrievers/#retrievers) as part of [application logic](https://python.langchain.com/docs/tutorials/rag/#orchestration) to retrieve the relevant data and generate output based on a specific template to be stored in MinIO

### Lab 1 - Indexing

To store the documents as embeddings, the [document_loader.py](document_loader.py) script should be sufficient
```bash
python document_loader.py
```
It assumes that the [Docker Compose Cluster](#docker-compose-cluster) is set up and the JSON files are copied into [MinIO](#minio) are in place.

Some references that were essential for implementing the loader are as follows:
* [LangChain > Tutorials > Build a Retrieval Augmented Generation (RAG) App: Part 1 > Indexing](https://python.langchain.com/docs/tutorials/rag/#indexing)
* [LangChain > Integrations > Components > Document loaders > AWS S3 File](https://python.langchain.com/docs/integrations/document_loaders/aws_s3_file/) - This document loader wasn't used because it's for handling unstructured data.
  * Ultimately the solution was to use `boto3` to read the content of the file and custom code to create the `Document` objects needed based on the [JSONLoader]( https://github.com/langchain-ai/langchain-community/blob/90860265dd6f0a9e840b8350ba8e8b2502225d51/libs/community/langchain_community/document_loaders/json_loader.py#L153-L168).
* [LangChain > Integrations > Components > Document loaders > JSONLoader](https://python.langchain.com/docs/integrations/document_loaders/json/) - This document loader wasn't used because it can't read from S3 path
  * [LangChain > How-to guides > How to load JSON](https://python.langchain.com/docs/how_to/document_loader_json/) - Some examples of how to use the document loader which would have been useful if the document loader had worked for our specific integration
  * [LangChain > How-to guides > How to split JSON data](https://python.langchain.com/docs/how_to/recursive_json_splitter/) - If the content is too large, then it's advisable to split it before storing into a knowledge base for RAG.
* [LangChain > Integrations > Components > Embedding models > OllamaEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/ollama/)
* [LangChain > Integrations > Components > Vector stores > PGVectorStore](https://python.langchain.com/docs/integrations/vectorstores/pgvectorstore/)

### Lab 1 - Retrieval and Generation

To query the documents and reason through the content, the [product_insight_agent.py](product_insight_agent.py) script is derived from [LangChain > Tutorials > Build a Retrieval Augmented Generation (RAG) App: Part 1 > Retrieval and Generation](https://python.langchain.com/docs/tutorials/rag/#orchestration) but modified to interact with the documents that were loaded.
```bash
python product_insight_agent.py
```

Query analysis that's referenced in the [RAG tutorial](https://python.langchain.com/docs/tutorials/rag/#query-analysis) is out of scope but an avenue to explore to allow the model to [rephrase the query](https://python.langchain.com/docs/concepts/retrieval/#query-analysis) for retrieval purposes.
