# derived from https://catalog.us-east-1.prod.workshops.aws/workshops/015a2de4-9522-4532-b2eb-639280dc31d8/en-US/20-agentcore-runtime/21-hosting-agent#getting-started
from strands import Agent, tool
from strands.agent import AgentResult
from strands_tools import calculator
import argparse
import json
from strands.models import Model
# from strands.models.bedrock import BedrockModel
from strands.models.ollama import OllamaModel
# Integration Point #1 to import the module
# from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Integration Point #2 to initialize the app
# app: BedrockAgentCoreApp = BedrockAgentCoreApp()

@tool
def weather():
    """ Get the weather """
    return "sunny"

# model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# model = BedrockModel(
#     model_id=model_id,
# )
model_id: str = 'llama3.2:3b'
model: Model = OllamaModel(
    host='http://localhost:11434',
    model_id=model_id
)
agent: Agent = Agent(
    model=model,
    tools=[calculator, weather],
    system_prompt="You're a helpful assistant. You can perform simple math calculations and tell the weather."
)

# Integration Point #3 to decorate invocation function
# @app.entrypoint
def strands_agent_bedrock(payload) -> str:
    """
    Invoke the agent with a payload
    """
    user_input: str = payload.get("prompt")
    result: AgentResult = agent(user_input)
    return result.message['content'][0]['text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response: str = strands_agent_bedrock(json.loads(args.payload))
    print(response)
    # Integration Point #4 to deploy the code as AgentCore Runtime, code above would be commented out
    # app.run()
