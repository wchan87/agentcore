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

## Tavily

Register with [Tavily Search API](https://app.tavily.com/home) to get `TAVILY_API_KEY` to set as environment variable.
```bash
export TAVILY_API_KEY=...
```
The JSON response structure for the search function is shared for reference if further improvements are to be made to the application code.
```json
{
  "query": ...,
  "follow_up_questions": ...,
  "answer": ...,
  "images": ...,
  "results": [
    {
      "url": ...,
      "title": ...,
      "content": ...,
      "score": ...,
      "raw_content": ...
    }
  ],
  "response_time": ...,
  "request_id": ... 
}
```

## Lab 1 - Product Insight Agent

[Lab 1: Product Insight Agent](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab1-single-agent) involves an agent that searches [Amazon Bedrock Knowledge Base](https://docs.aws.amazon.com/en_us/bedrock/latest/userguide/kb-how-data.html) (i.e., [Retrieval-Augmented Generation (RAG)](https://python.langchain.com/docs/tutorials/rag/)) that's populated via [documents uploaded](https://docs.aws.amazon.com/en_us/bedrock/latest/userguide/kb-data-source-customize-ingestion.html) to S3 bucket which is [chunked](https://docs.aws.amazon.com/en_us/bedrock/latest/userguide/kb-data-source-customize-ingestion.html), converted to embeddings and summarized into a report stored back into S3 bucket with the steps as follows:
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

**Note:** Query analysis that's referenced in the [RAG tutorial](https://python.langchain.com/docs/tutorials/rag/#query-analysis) is out of scope but an avenue to explore to allow the model to [rephrase the query](https://python.langchain.com/docs/concepts/retrieval/#query-analysis) for retrieval purposes. A pre-processing step was added in the `extract` function to narrow down the main vehicle in scope for the request. The pre-processing step will be compared to the approach is documented by LangChain in the future.

**Note:** Tutorial shows another approach to retrieve canned prompt that stored in LangSmith under [rlm/rag-prompt](https://smith.langchain.com/hub/rlm/rag-prompt).
```python
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

prompt: ChatPromptTemplate = hub.pull('rlm/rag-prompt')
```

## Lab 2 - Market Analyst Agent

[Lab 2: Market Analyst Agent](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab2-market-agent) involves an agent to interact with external tools (i.e., APIs) and summarized into a report back into S3 bucket based on the steps below:
1. Extract the core product/trend specified the question
2. Iterate through 4 specific queries against Tavily regarding the core product/trend
3. Summarize the search results from the prior step into the standardized format
4. Persist the standardized format to S3 bucket and return the S3 location

The steps are inferred from the [Instructions for the Agent](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab2-market-agent/step-02-agent-check) that was partially provided.

Ensure that `TAVILY_API_KEY` is [set](#tavily) before running the [market_analysis_agent.py](market_analysis_agent.py) script:
```bash
python market_analysis_agent.py
```

## Lab 3 - Reporter Agent

[Lab 3: Reporter Agent](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab3-reporter-agent) doesn't have enough context to enable implementation aside from it combining the results in S3 from prior two agents and generating a comprehensive report.

## Lab 4 - Supervisor Agent

[Lab 4: Multi-Agent Collaboration Setup](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US/40-hands-on-lab/lab5-mac) involves a supervisor agent that interacts with the previous agents to orchestrate the following workflow:
1. In parallel, execute the [product insight agent](#lab-1---product-insight-agent) and [market analysis agent](#lab-2---market-analyst-agent) to generate reports
2. Wait for both agents to complete and verify the reports are generated
3. Dispatch the [reporter agent](#lab-3---reporter-agent) to aggregate the reports and generate HTML to address the question from the user
4. Return final HTML report to the user

The documentation on the workshop isn't particularly complete regarding the reporter and supervisor agents but the variant workshop, [Amazon Bedrock multi-agent collaboration](https://catalog.us-east-1.prod.workshops.aws/workshops/1031afa5-be84-4a6a-9886-4e19ce67b9c2/en-US) may be worth checking as it has sample code, [aws-samples/bedrock-multi-agents-collaboration-workshop](https://github.com/aws-samples/bedrock-multi-agents-collaboration-workshop) that is accessible.

References for the multi-agent collaboration feature in Amazon Bedrock are included for completeness but would be difficult to simulate locally:
* [AWS Blogs > Introducing multi-agent collaboration capability for Amazon Bedrock (preview)](https://aws.amazon.com/blogs/aws/introducing-multi-agent-collaboration-capability-for-amazon-bedrock/) - Released 2024-12-03
* [AWS Blogs > Amazon Bedrock announces general availability of multi-agent collaboration](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-announces-general-availability-of-multi-agent-collaboration/) - Released 2025-03-10
* [AWS > Documentation > Amazon Bedrock > User Guide > Use multi-agent collaboration with Amazon Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-multi-agent-collaboration.html) - Possible counterpart to the workshop sample code if this feature is needed

To simulate multi-agent collaboration locally, the following resources appear to be relevant
* [LangGraph > Examples > Agent Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
  * [GitHub > langchain-ai/langgraph > Tutorials > Multi-agent supervisor](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.md) - Source code for above
* [PyPI > langgraph-supervisor](https://pypi.org/project/langgraph-supervisor/) - Python module that may be required to implement this workflow
  * [langchain-ai/langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py) - Source code for above
* [LangGraph > Get started > General concepts > Workflows and Agents](https://langchain-ai.github.io/langgraph/tutorials/workflows/)

Alternatively, [Agent2Agent (A2A) and Model Context Protocol (MCP)](https://a2a-protocol.org/dev/topics/a2a-and-mcp/) are other avenues of research to accomplish the coordination needed.

The first script, [supervisor_agent.py](supervisor_agent.py) represents using the `langgraph-supervisor` to wire together the orchestration based on [worker agents](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/#1-create-worker-agents) and [supervisor agent](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/#2-create-supervisor-with-langgraph-supervisor). **Note:** It has a dependency on [TAVILY_API_KEY](#tavily) being defined as environment variable to work.
```bash
python supervisor_agent.py
```
