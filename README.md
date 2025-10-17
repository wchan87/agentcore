# AgentCore

* AWS Workshops Reviewed
  * [Diving Deep into Bedrock AgentCore](diving-deep-into-bedrock-agentcore.md)
  * [Amazon Bedrock Multi-Agent Collaboration](bedrock-multi-agent-collaboration.md)
    * [Amazon Bedrock multi-agent collaboration](https://catalog.us-east-1.prod.workshops.aws/workshops/1031afa5-be84-4a6a-9886-4e19ce67b9c2/en-US) - A variant which was recently discovered along with some sample code, [aws-samples/bedrock-multi-agents-collaboration-workshop](https://github.com/aws-samples/bedrock-multi-agents-collaboration-workshop)
* Other AWS Workshops
  * [Amazon Bedrock AgentCore Workshop: From Basics to Advanced Agent Development](https://catalog.us-east-1.prod.workshops.aws/workshops/abd92795-9a36-4e63-a115-ad04f483248c/en-US) - A deeper dive into the AgentCore Runtime

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

**Note:** The workshops have a new Python package and project manager, [uv](https://docs.astral.sh/uv/) that may be considered for usage.

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
* Use [Docker Desktop Model Runner](https://docs.docker.com/ai/model-runner/) (see [blog post for more details](https://www.docker.com/blog/run-llms-locally/))
  * Enable Model Runner
      ```bash
      docker desktop enable model-runner --cors all --gpu enable --tcp=12434
      ```
  * Pull a model
      ```bash
      docker model pull ai/gpt-oss:latest
      ```
  * Run a model
    * Run a model via CLI
      ```bash
      docker model run ai/gpt-oss:latest "Explain Docker in one sentence"
      ```
    * Run a model with frameworks. Model Runner has Open AI compatible endpoints
      * Base URL 
        * `http://localhost:12434/engines/v1` (if from the host)
        * `http://model-runner.docker.internal/engines/v1` (if from another Docker container itself)
      * API Key - any placeholder like "docker" works
      * Model - image name which you pulled the image as (ex: `ai/gpt-oss:latest`)
    * Frameworks
      * [LangChain > Components > Chat models > OpenAI](https://python.langchain.com/docs/integrations/chat/openai/)
        ```bash
        pip install -qU langchain-openai
        ```
        ```bash
        python -c 'from langchain_openai import ChatOpenAI; llm = ChatOpenAI(model="ai/gpt-oss:latest", api_key="docker", base_url="http://localhost:12434/engines/v1"); messages = [("system", "You are a helpful AI assistant"), ("human", "Explain Docker in one sentence")]; ai_msg = llm.invoke(messages); print(ai_msg.content)'
        ```
      * [Strands Agents > User Guide > Model Providers > OpenAI](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/openai/)
         ```bash
         pip install -qU 'strands-agents[openai]'
         ```
         ```bash
         python -c 'from strands import Agent; from strands.models.openai import OpenAIModel; model = OpenAIModel(client_args={"api_key":"docker", "base_url":"http://localhost:12434/engines/v1"}, model_id="ai/gpt-oss:latest"); agent = Agent(model=model); response = agent("Explain Docker in one sentence"); print(response)'
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
