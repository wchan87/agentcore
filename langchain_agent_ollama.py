# derived from https://github.com/awslabs/amazon-bedrock-agentcore-samples/blob/main/01-tutorials/01-AgentCore-runtime/01-hosting-agent/02-langgraph-with-bedrock-model/runtime_with_langgraph_and_bedrock_models.ipynb
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import argparse
import json
import operator
import math
# from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_ollama import ChatOllama
# Integration Point #1 to import the module
# from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Integration Point #2 to initialize the app
# app: BedrockAgentCoreApp = BedrockAgentCoreApp()

# Create calculator tool
@tool
def calculator(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.

    Args:
        expression: A mathematical expression as a string (e.g., "2 + 3 * 4", "sqrt(16)", "sin(pi/2)")

    Returns:
        The result of the calculation as a string
    """
    try:
        # Define safe functions that can be used in expressions
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
            # Math functions
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
            "degrees": math.degrees, "radians": math.radians,
            # Basic operators (for explicit use)
            "add": operator.add, "sub": operator.sub,
            "mul": operator.mul, "truediv": operator.truediv,
        }

        # Evaluate the expression safely
        result = eval(expression, safe_dict)
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {str(e)}"
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except Exception as e:
        return f"Error: {str(e)}"


# Create a custom weather tool
@tool
def weather():
    """Get weather"""  # Dummy implementation
    return "sunny"


# Define the agent using manual LangGraph construction
def create_agent():
    """Create and configure the LangGraph agent"""
    # Initialize your LLM (adjust model and parameters as needed)
    # model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    # llm: BaseChatModel = ChatBedrock(
    #     model=model_id, # or your preferred model
    #     model_kwargs={"temperature": 0.1}
    # )
    model_id: str = 'llama3.2:3b'
    llm: BaseChatModel = ChatOllama(
        base_url='http://localhost:11434',
        model=model_id
    )

    # Bind tools to the LLM
    tools = [calculator, weather]
    llm_with_tools = llm.bind_tools(tools)

    # System message
    system_message = "You're a helpful assistant. You can do simple math calculation, and tell the weather."

    # Define the chatbot node
    def chatbot(state: MessagesState):
        # Add system message if not already present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_message)] + messages

        result = llm_with_tools.invoke(messages)
        return {"messages": [result]}

    # Create the graph
    graph_builder = StateGraph(MessagesState)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools))

    # Add edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")

    # Set entry point
    graph_builder.set_entry_point("chatbot")

    # Compile the graph
    return graph_builder.compile()

# Initialize the agent
agent = create_agent()

# Integration Point #3 to decorate invocation function
# @app.entrypoint
def langgraph_bedrock(payload) -> str:
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")

    # Create the input in the format expected by LangGraph
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    # Extract the final message content
    return result["messages"][-1].content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response: str = langgraph_bedrock(json.loads(args.payload))
    print(response)
    # Integration Point #4 to deploy the code as AgentCore Runtime, code above would be commented out
    # app.run()
