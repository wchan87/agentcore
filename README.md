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

## Frameworks

### Strands Agents

See [Strands Agents](https://strandsagents.com/) for more information
* [Strands - Ollama](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/ollama/)
* [Strands - llama.cpp](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/llamacpp/)

### LangChain and LangGraph

See [LangChain> LangGraph](https://www.langchain.com/langgraph) for more information
* [LangChain - ChatOllama](https://python.langchain.com/docs/integrations/chat/ollama/)
* [LangChain - Llama.cpp](https://python.langchain.com/docs/integrations/chat/llamacpp/)
