# Diving Deep into Bedrock AgentCore

Based on [Diving Deep into Bedrock AgentCore](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US)

## AgentCore Fundamentals

Description for the components of AgentCore are from [Amazon Bedrock AgentCore Fundamentals](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/10-fundamentals)

* [AgentCore Runtime](#agentcore-runtime)
  > AgentCore Runtime is a secure, serverless runtime purpose-built for deploying and scaling dynamic AI agents and tools using any open-source framework (including Strands Agents, LangGraph, and CrewAI), any protocol, and any model. Runtime was built to work for agentic workloads with industry-leading extended runtime support, fast cold starts, true session isolation, built-in identity, and support for multi-modal payloads. Developers can focus on innovation while Amazon Bedrock AgentCore Runtime handles infrastructure and security—accelerating time-to-market.
  * Framework (i.e., [Strands Agents](https://strandsagents.com/), [LangGraph](https://www.langchain.com/langgraph), [CrewAI](https://www.crewai.com/) and [LlamaIndex](https://www.llamaindex.ai/))
    * Agent instructions
    * Agent local tools
    * Agent context
* Models
* [AgentCore Memory](#agentcore-memory)
    > AgentCore Memory makes it easy for developers to build context aware agents by eliminating complex memory infrastructure management while providing full control over what the AI agent remembers. Memory provides industry-leading accuracy along with support for both short-term memory for multi-turn conversations and long-term memory that can be shared across agents and sessions.
* [AgentCore Gateway](#agentcore-gateway)
  > AgentCore Gateway provides a secure way for agents to discover and use tools along with easy transformation of APIs, Lambda functions, and existing services into agent-compatible tools. Gateway eliminates weeks of custom code development, infrastructure provisioning, and security implementation so developers can focus on building innovative agent applications. AgentCore Gateway's powerful built-in semantic search capability helps agents effectively search tools to find the most appropriate ones for specific contexts, allowing agents to take advantage of thousands of tools while minimizing prompt size and reducing latency.
* [AgentCore Browser](#agentcore-browser)
* [AgentCore Code Interpreter](#agentcore-code-interpreter)
* [AgentCore Identity](#agentcore-identity)
  > AgentCore Identity provides a secure, scalable agent identity and access management capability accelerating AI agent development. It is compatible with existing identity providers, eliminating needs for user migration or rebuilding authentication flows. AgentCore Identity's helps to minimize consent fatigue with a secure token vault and allows you to build streamlined AI agent experiences. Just-enough access and secure permission delegation allow agents to securely access AWS resources and third-party tools and services.
* CloudWatch GenAI Observability
    > AgentCore Observability helps developers trace, debug, and monitor agent performance in production through unified operational dashboards. With support for OpenTelemetry compatible telemetry and detailed visualizations of each step of the agent workflow, AgentCore enables developers to easily gain visibility into agent behavior and maintain quality standards at scale.

## AgentCore Runtime

### AgentCore Runtime Agent

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

### AgentCore Runtime MCP Server

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

## AgentCore Gateway

[AgentCore Gateway](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/30-agentcore-gateway) functions as if it is a MCP server itself and map tool definitions to other resources such as:
* REST APIs that implement OpenAPI schema
* AWS Lambdas (this appears to be the primary rationale for it to enable existing functionalities to be transformed into MCP servers)
* Interfaces that adhere to the [Smithy model](https://smithy.io/index.html)

AgentCore Gateway handles the following
* outbound authentication contract with each target
* translates the communication with the target to MCP JSON-RPC
* ensure security is enforced between inbound identities and outbound scopes (i.e., the user is authorized to make the call to the relevant tool with the correct outbound authentication)
* providing natural-language query to allow the agent to select the right tool for its context

The agent running the MCP client is expected to obtain access token from the OIDC provider that is registered when the AgentCore Gateway is created (i.e., OAuth authorizer configuration).

## AgentCore Identity

[AgentCore Identity](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/40-agentcore-identity) ensures that the agents acting independently or on behalf of users has resource access checked and enforced on the target tool resources or gateways.

[Inbound authentication](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/40-agentcore-identity/41-inbound-auth) enables the following:
> In production environments, AI agents often handle sensitive data and perform critical business operations. Without proper authentication:
> * Unauthorized access could lead to data breaches or misuse of agent capabilities
> * Lack of audit trails makes it difficult to track who accessed what resources
> * No user context prevents agents from providing personalized experiences
> * Security compliance requirements cannot be met

To support inbound authentication, the original AgentCore Runtime configuration that was previously setup is [enhanced](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/40-agentcore-identity/41-inbound-auth#update-agentcore-runtime-deployment) to include `authorizer_configuration` as well. The corresponding client code must now be enhanced to pass `bearer_token` corresponding to the user.

[Outbound authentication](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/40-agentcore-identity/42-outbound-auth) enables the following:
> Modern AI agents need to interact with multiple external services to provide comprehensive functionality. Without proper Outbound Auth:
> * Credential sprawl leads to security vulnerabilities and management overhead
> * Hard-coded secrets in agent code create security risks
> * Manual token management is error-prone and doesn't scale
> * Lack of centralized control makes credential rotation and auditing difficult
> * User consent fatigue degrades user experience with repeated authorization prompts

Outbound authentication enables "OAuth 2LO/3LO or API keys" to validate access to the target resources. 2LO and 3LO means "2 legged OAuth" and "3 legged OAuth" respectively with an explanation [here](https://lekkimworld.com/2020/03/12/2-legged-vs-3-legged-oauth/).

## AgentCore Memory

[AgentCore Memory](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/50-agentcore-memory) provides short-term and long-term memory:
> AgentCore Memory lets your AI agents deliver intelligent, context-aware, and personalized interactions by maintaining both immediate and long-term knowledge. AgentCore Memory offers two types of memory:
> * Short-term memory: Stores conversations to keep track of immediate context.
> * Long-term memory: Stores extracted insights - such as user preferences, semantic facts, and summaries - for knowledge retention across sessions.

In addition, the challenges that it's attempting to solve are
> Traditional AI agents face several critical challenges that AgentCore Memory addresses:
> * Infrastructure management: AgentCore Memory provides a serverless solution with a single API that automatically manages the underlying infrastructure.
> * Context window limits: AgentCore Memory persists the interaction history once and lets the agent selectively retrieve only what is relevant.
> * Cross‑session continuity: AgentCore Memory's long‑term tier retains user preferences, semantic facts, and summaries so returning users experience continuity without repeating themselves.
> * Reliable consistency: AgentCore Memory provides a managed schema (events and extracted records), asynchronous consolidation, namespace scoping, and role‑based access controls out of the box.
> * Namespace isolation: Enterprise deployments require per‑user isolation, PII protection, and time‑bound retention. AgentCore Memory supports hierarchical namespaces (`/strategy/{strategyId}/actor/{actorId}/session/{sessionId}`), encrypted storage with customer‑managed KMS keys, event expiry settings up to 365 days, and IAM context keys for least‑privilege policies.
> * Information processing and persistence: Built-in and custom strategies intelligently capture main concepts from interactions and persists them. These strategy configurations help extract and persist relevant information without requiring custom parsing code, while supporting both immediate context and processed information across sessions.

For handling [short-term memory](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/50-agentcore-memory/51-short-term-memory), there are two approaches:
* Use hooks to store conversation (user/agent) events as they happen and also retrieve prior conversation history on agent being initialized
* Create a tool to manage the memory instead and let the agent make the decision on storage and retrieval

The former is recommended without directly modifying the core code for the agent.

For handling [long-term memory](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/50-agentcore-memory/52-long-term-memory), persistent information are organized as follows:
> Namespaces organize memory records within strategies using a path-like structure. They can include variables that are dynamically replaced:
> * `support/facts/{sessionId}`: Organizes facts by session
> * `user/{actorId}/preferences`: Stores user preferences by actor ID
> * `meetings/{memoryId}/summaries/{sessionId}`: Groups summaries by memory
> 
> The `{actorId}`, `{sessionId}`, and `{memoryId}` variables are automatically replaced with actual values when storing and retrieving memories.

There are four memory strategies supported:
* [Semantic Memory Strategy](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/50-agentcore-memory/52-long-term-memory#1.-semantic-memory-strategy) - "Stores factual information extracted from conversations using vector embeddings for similarity search."
* [Session Summary Memory Strategy](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/50-agentcore-memory/52-long-term-memory#2.-session-summary-memory-strategy) - "Creates and maintains summaries of conversations to preserve context for long interactions. The sessionId parameter is mandatory for this strategy."
* [User Preference Memory Strategy](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/50-agentcore-memory/52-long-term-memory#3.-user-preference-memory-strategy) - "Tracks user-specific preferences and settings to personalize interactions."
* [Custom Memory Strategy](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/50-agentcore-memory/52-long-term-memory#4.-custom-memory-strategy) - "Allows customization of prompts for extraction and consolidation, providing flexibility for specialized use cases."

An open question is whether the knowledge accumulated could be leveraged to provide anonymous and aggregated feedback to improve the collective knowledge base used by other agents such as [Amazon Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html). In other words, can the agent's long-term memory help to improve the domain knowledge if sufficient care is taken to anonymize and aggregate the personal interactions.

## AgentCore Tools

[AgentCore Tools](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/60-agentcore-tools) appears to encapsulate two different capabilities:
* [Code Interpreter](#agentcore-code-interpreter) - "A secure environment for executing code in multiple languages and processing data."
* [Browser Tool](#agentcore-browser) - "A managed service for web interactions, enabling AI agents to navigate and interact with websites."

### AgentCore Code Interpreter

[AgentCore Code Interpreter](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/60-agentcore-tools/61-code-interpreter) appears to be a mechanism for users that have data or information that they can run limited analytics/data processing code that the LLM can generate from natural language.

### AgentCore Browser

[AgentCore Browser](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/60-agentcore-tools/62-browser-tool) appears to be a mechanism for the agent to be able to act if there is no API but a website using a headless browser and tools like [Playwright](https://playwright.dev/). A screenshot is shared with the agent and the user to provide the necessary information from the browser. It seems that through this interaction, CAPTCHAS could be solved by the user for human intervention.

## AgentCore Observability

[AgentCore Observability](https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/70-agentcore-observability) provides the means "to trace, debug, and monitor agent performance in production environments" due to [OTEL](https://opentelemetry.io/) logs generated by the components.

The [aws-opentelemetry-distro](https://pypi.org/project/aws-opentelemetry-distro/) seems to be a necessary dependency which automatically adds the instrumentation when the Docker image is generated for the [AgentCore Runtime](#agentcore-runtime).
