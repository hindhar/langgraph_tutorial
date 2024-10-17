import os
import getpass
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from IPython.display import Image, display

# Load environment variables from .env file
load_dotenv()

# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create the StateGraph
graph_builder = StateGraph(State)

# Set up the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Define the chatbot function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Visualize the graph (optional)
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    print("Graph visualization failed. This requires extra dependencies.")

# Function to stream graph updates
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Main chat loop
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
