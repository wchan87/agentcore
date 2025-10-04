# AgentCore

* AWS Workshops
  * [Diving Deep into Bedrock AgentCore](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US)
  * [Amazon Bedrock AgentCore Workshop: From Basics to Advanced Agent Development](https://catalog.us-east-1.prod.workshops.aws/workshops/abd92795-9a36-4e63-a115-ad04f483248c/en-US)
  * [Amazon Bedrock multi-agent collaboration: Building JARVIS (Smart Assistant) to solve complex problems with simple AI Agents](https://catalog.us-east-1.prod.workshops.aws/workshops/c68a2fb4-8b25-480f-ab0b-129778f96d4d/en-US)

## Python Virtual Environment

1. Setup virtual environment
    ```bash
    python -m venv .venv
    ```
2. Activate virtual environment
    ```bash
    source .venv/Scripts/activate
    ```
3. Install dependencies
    ```bash
    pip install -vr requirements-dev.txt
    ```

## Local LLMs

* [LM Studio](https://lmstudio.ai/) can be used to run LLMs locally but it's not straightforward on whether it integrates with Python for the other frameworks to invoke. See [Python SDK](https://lmstudio.ai/docs/python) for more information
* [llama.cpp](https://github.com/ggml-org/llama.cpp)
  * [Docker instructions for llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md) - It's not entirely why the LLaMA models have to be converted to [ggml](https://huggingface.co/blog/introduction-to-ggml) to be runnable via the Docker container.
    ```bash
    docker pull ghcr.io/ggml-org/llama.cpp:full-cuda
    ```
  * Pre-built binaries can be found under [releases](https://github.com/ggml-org/llama.cpp/releases) as an alternative.
* [ollama](https://ollama.com/)
  * [DockerHub > ollama/ollama](https://hub.docker.com/r/ollama/ollama)
    * Pull the image
        ```bash
        docker pull ollama/ollama
        ```
    * Start the container to use the GPUs with name volumne, `ollama` containing the models
        ```bash
        docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
        ```
    * Run a model to get it downloaded by the container, [gemma3](https://ollama.com/library/gemma3) in this example. Another model is [llama3.2](https://ollama.com/library/llama3.2) which supports tools as per [Ollama Blog > Tool support](https://ollama.com/blog/tool-support)
        ```bash
        docker exec -it ollama ollama run gemma3:4b
        docker exec -it ollama ollama run llama3.2:3b
        ```

## Frameworks

### Strands Agents

See [Strands Agents](https://strandsagents.com/) for more information
* [Strands - Ollama](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/ollama/)
* [Strands - llama.cpp](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/llamacpp/)

### LangChain and LangGraph

See [LangChain> LangGraph](https://www.langchain.com/langgraph) for more information
* [LangChain - ChatOllama](https://python.langchain.com/docs/integrations/chat/ollama/)
* [LangChain - Llama.cpp](https://python.langchain.com/docs/integrations/chat/llamacpp/)

## Diving Deep into Bedrock AgentCore

### AgentCore Fundamentals

Description for the components of AgentCore are from [Amazon Bedrock AgentCore Fundamentals](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/10-fundamentals)

* AgentCore Runtime
  > AgentCore Runtime is a secure, serverless runtime purpose-built for deploying and scaling dynamic AI agents and tools using any open-source framework (including Strands Agents, LangGraph, and CrewAI), any protocol, and any model. Runtime was built to work for agentic workloads with industry-leading extended runtime support, fast cold starts, true session isolation, built-in identity, and support for multi-modal payloads. Developers can focus on innovation while Amazon Bedrock AgentCore Runtime handles infrastructure and security—accelerating time-to-market.
  * Framework (i.e., [Strands Agents](https://strandsagents.com/), [LangGraph](https://www.langchain.com/langgraph), [CrewAI](https://www.crewai.com/) and [LlamaIndex](https://www.llamaindex.ai/))
    * Agent instructions
    * Agent local tools
    * Agent context
* Models
* AgentCore Memory
    > AgentCore Memory makes it easy for developers to build context aware agents by eliminating complex memory infrastructure management while providing full control over what the AI agent remembers. Memory provides industry-leading accuracy along with support for both short-term memory for multi-turn conversations and long-term memory that can be shared across agents and sessions.
* AgentCore Gateway
  > AgentCore Gateway provides a secure way for agents to discover and use tools along with easy transformation of APIs, Lambda functions, and existing services into agent-compatible tools. Gateway eliminates weeks of custom code development, infrastructure provisioning, and security implementation so developers can focus on building innovative agent applications. AgentCore Gateway's powerful built-in semantic search capability helps agents effectively search tools to find the most appropriate ones for specific contexts, allowing agents to take advantage of thousands of tools while minimizing prompt size and reducing latency.
* AgentCore Browser
* AgentCore Code Interpreter
* AgentCore Identity
  > AgentCore Identity provides a secure, scalable agent identity and access management capability accelerating AI agent development. It is compatible with existing identity providers, eliminating needs for user migration or rebuilding authentication flows. AgentCore Identity's helps to minimize consent fatigue with a secure token vault and allows you to build streamlined AI agent experiences. Just-enough access and secure permission delegation allow agents to securely access AWS resources and third-party tools and services.
* CloudWatch GenAI Observability
    > AgentCore Observability helps developers trace, debug, and monitor agent performance in production through unified operational dashboards. With support for OpenTelemetry compatible telemetry and detailed visualizations of each step of the agent workflow, AgentCore enables developers to easily gain visibility into agent behavior and maintain quality standards at scale.

### AgentCore Runtime

#### AgentCore Runtime Agent

In order to integrate with AgentCore Runtime, [bedrock-agentcore](https://github.com/aws/bedrock-agentcore-sdk-python) needs to be installed.

Using [Strands Agents](#strands-agents) framework, [strands_agent_ollama.py](strands_agent_ollama.py) is based on [sample code](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/21-hosting-agent#getting-started) and is invoked as follows:
```bash
python strands_agent_ollama.py "{\"prompt\": \"What's 1+1?\"}"
python strands_agent_ollama.py "{\"prompt\": \"What is the weather?\"}"
```

Using [LangChain and LangGraph](#langchain-and-langgraph), [langchain_agent_ollama.py](langchain_agent_ollama.py) is based on [sample code](https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/01-AgentCore-runtime/01-hosting-agent/02-langgraph-with-bedrock-model/runtime_with_langgraph_and_bedrock_models.ipynb) and prior example to confirm `langchain` also works as well:
```bash
python langchain_agent_ollama.py "{\"prompt\": \"1 + 1\"}"
python langchain_agent_ollama.py "{\"prompt\": \"What is the weather?\"}"
```

There are additional steps to make the code usable via AgentCore Runtime such as
1. [Make the four changes](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/21-hosting-agent#prepare-the-agent-for-agentcore-runtime) commented out in the code to integrate with AgentCore.
2. Configure the AgentCore Runtime which functionally packages the code into Docker container on Amazon ECR repository.
3. Launch the AgentCore Runtime. There are commands to check its status referenced by GitHub sample codes
   * [runtime_with_strands_and_bedrock_models.ipynb](https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/01-AgentCore-runtime/01-hosting-agent/01-strands-with-bedrock-model/runtime_with_strands_and_bedrock_models.ipynb)
   * [runtime_with_langgraph_and_bedrock_models.ipynb](https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/01-AgentCore-runtime/01-hosting-agent/02-langgraph-with-bedrock-model/runtime_with_langgraph_and_bedrock_models.ipynb)
   * [runtime_with_strands_and_openai_models.ipynb](https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/01-AgentCore-runtime/01-hosting-agent/03-strands-with-openai-model/runtime_with_strands_and_openai_models.ipynb)
4. [Invoke](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/21-hosting-agent#invoke-your-agent) the AgentCore Runtime via boto3.

#### AgentCore Runtime MCP Server

Based on [server sample code](https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/01-AgentCore-runtime/02-hosting-MCP-server/hosting_mcp_server.ipynb), start the MCP server which uses [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http) via a `http://localhost:8080/mcp` endpoint to support client requests:
```bash
python mcp_server.py &
```
The process id returned from the prior command can be used to kill the process afterwards:
```bash
kill <pid>
```
The code does the following for reference:
> * FastMCP: Creates an MCP server that can host your tools
> * @mcp.tool(): Decorator that turns your Python functions into MCP tools
> * stateless_http=True: Required for AgentCore Runtime compatibility
> * Tools: Three simple tools demonstrating different types of operations

The corresponding [client sample code](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/22-hosting-mcp-server#create-local-testing-client) just checks to confirm the MCP tools are exposed.
```bash
python my_mcp_client.py
```
```
INFO:     127.0.0.1:53603 - "POST /mcp HTTP/1.1" 200 OK
[10/04/25 18:25:49] INFO     Terminating session: None    streamable_http.py:630
INFO:     127.0.0.1:53605 - "POST /mcp HTTP/1.1" 202 Accepted
                    INFO     Terminating session: None    streamable_http.py:630
INFO:     127.0.0.1:53607 - "POST /mcp HTTP/1.1" 200 OK
[10/04/25 18:25:50] INFO     Processing request of type            server.py:664
                             ListToolsRequest
Available tools:
  - add_numbers: Add two numbers together
  - multiply_numbers: Multiply two numbers together
  - greet_user: Greet a user by name
                    INFO     Terminating session: None    streamable_http.py:630
```

There are additional steps in order to setup the MCP Server on AgentCore Runtime
1. [Setting up Amazon Cognito](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/22-hosting-mcp-server#setting-up-amazon-cognito-for-authentication) to allow it to provide JWT tokens to access the MCP server. The setup of Amazon Cognito is driven by the sample code, [utils.py](https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/utils.py)
2. [Creating IAM role](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/22-hosting-mcp-server#create-iam-execution-role-for-the-agentcore-runtime) for the MCP server.
3. [Configure AgentCore Runtime](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/22-hosting-mcp-server#configuring-agentcore-runtime-deployment) to ensure that it has the source code, [mcp_server.py](mcp_server.py) and `requirements.txt` that's needed to run the code along with the Amazon Cognito configuration from prior step.
4. Store "the Agent ARN and Cognito configuration in AWS Systems Manager Parameter Store and AWS Secrets Manager for easy retrieval"
5. [Invoke the MCP Server](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/22-hosting-mcp-server#creating-remote-testing-client) remotely by passing the prior setup information.
6. Actual usage of the tools on MCP server is covered in [Step 2](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/22-hosting-mcp-server#testing) and appears to invoke [mcp.ClientSession.call_tool](https://github.com/modelcontextprotocol/python-sdk/blob/814c9c024a86fa0f608e87b15b21a0a16a926d61/src/mcp/client/session.py#L270-L296) function.

Advanced topics for AgentCore Runtime are covered as
* [Streaming Responses with AgentCore Runtime](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/23-advanced-concepts#streaming-responses-with-agentcore-runtime) - This topic is relevant if large content or significant processing time is required.
* [Session and Context Management](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/23-advanced-concepts#session-and-context-management) - This topic is relevant if context/state has to be shared across multiple invocation.
   > Within a session, AgentCore Runtime maintains:
   > * Conversation History: Previous interactions and responses
   > * Application State: Variables and objects created during execution
   > * File System: Any files created or modified during the session
   > * Environment Variables: Custom settings and configurations
* [Handling Large Multi-Modal Payloads](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/23-advanced-concepts#handling-large-multi-modal-payloads) - This topic is relevant if multi-modal content (i.e., binary) up to 100 MB have to be handled.
