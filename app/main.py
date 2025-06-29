from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import json
import os

# --- Config & LLM setup ---

def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        # Create a default config if file doesn't exist
        default_config = {
            "model1_summarize": {
                "model": "gemma3:12b",
                "temperature": 0.7
            },
            "model2_points": {
                "model": "gemma3:12b", 
                "temperature": 0.3
            }
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config
    
    with open(config_path, "r") as f:
        return json.load(f)

def get_llm(config: dict, file_key: str):
    """Get LLM instance from config"""
    settings = config.get(file_key, {"model": "gemma3:12b", "temperature": 0.7})
    return ChatOllama(
        model=settings["model"],
        temperature=settings.get("temperature", 0.7)
    )

def load_prompt_template(template_path, default_prompt):
    """Load prompt template with fallback"""
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            return f.read()
    else:
        return default_prompt

# Load configuration and models
try:
    config = load_config("app/config.json")
    
    # Model 1: Paragraph Summarizer (Main Chatbot)
    model1_prompt = load_prompt_template(
        "app/model1_summarize.txt",
        """You are a summarization assistant. Your task is to take the input text and create a detailed paragraph summary. 
        Focus on capturing the main ideas and important details in a coherent paragraph format.

        Input: {input}

        Provide a comprehensive paragraph summary:"""
    )
    
    # Model 2: Point Generator (Tool)
    model2_prompt = load_prompt_template(
        "app/model2_points.txt", 
        """You are a point extraction assistant. Take the paragraph summary and convert it into clear, concise bullet points.
        Extract the key information and present it in a short, structured format.

        Summary: {input}

        Convert to clear points in short format:"""
    )
    
    model1_llm = get_llm(config, "model1_summarize")  # Main chatbot
    model2_llm = get_llm(config, "model2_points")     # Tool model
    
except Exception as e:
    print(f"Warning: Configuration loading failed: {e}")
    # Fallback configuration
    model1_llm = ChatOllama(model="gemma3:12b", temperature=0.7)
    model2_llm = ChatOllama(model="gemma3:12b", temperature=0.3)
    model1_prompt = "Summarize the following input into a comprehensive paragraph: {input}"
    model2_prompt = "Convert the following summary into clear bullet points: {input}"

# --- Tool Definition ---

@tool
def generate_points(summary_text: str) -> str:
    """Tool: Second model that converts paragraph summary to clear points"""
    try:
        prompt = model2_prompt.format(input=summary_text)
        response = model2_llm.invoke(prompt)
        
        # Handle different response types
        if isinstance(response, dict):
            return response.get("content", str(response))
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)
    except Exception as e:
        return f"Error in generate_points: {str(e)}"

# Only one tool - the second model
tools = [generate_points]

# --- Bind tools to Model 1 (Main Chatbot) ---
model1_with_tools = model1_llm.bind_tools(tools)

# --- LangGraph Setup ---

class State(TypedDict):
    messages: Annotated[list, add_messages]

def main_chatbot_node(state):
    """Model 1: Main chatbot that summarizes and can call the points tool"""
    try:
        # Get the latest message
        latest_message = state["messages"][-1]
        
        # If it's the initial input, create summary
        if isinstance(latest_message, HumanMessage):
            prompt = model1_prompt.format(input=latest_message.content)
            summary_response = model1_llm.invoke(prompt)
            
            # Extract summary content
            if hasattr(summary_response, "content"):
                summary_content = summary_response.content
            else:
                summary_content = str(summary_response)
            
            # Now call the tool to get points
            points_result = generate_points.invoke({"summary_text": summary_content})
            
            # Model 1 reviews the points and creates final output
            final_prompt = f"""Original Input: {latest_message.content}

Summary Created: {summary_content}

Points Generated: {points_result}

Now provide the final refined output combining the summary and points:"""
            
            final_response = model1_llm.invoke(final_prompt)
            return {"messages": [final_response]}
        
        else:
            # For subsequent messages, use model1 with tools
            return {"messages": [model1_with_tools.invoke(state["messages"])]}
            
    except Exception as e:
        error_msg = f"Error in main_chatbot_node: {str(e)}"
        return {"messages": [error_msg]}

# Create graph - Simplified as per your flow
graph_builder = StateGraph(State)

# Add only the main chatbot node (Model 1)
# Model 2 is called as a tool within Model 1
graph_builder.add_node("main_chatbot", main_chatbot_node)

# Direct flow: START -> main_chatbot -> END
graph_builder.add_edge(START, "main_chatbot")
graph_builder.add_edge("main_chatbot", END)

# Compile graph
graph = graph_builder.compile()

# --- Graph Visualization ---
def save_graph_image():
    """Save graph visualization"""
    try:
        os.makedirs("output", exist_ok=True)
        image_data = graph.get_graph().draw_mermaid_png()
        output_path = "output/agentic_flow_graph.png"

        with open(output_path, "wb") as f:
            f.write(image_data)

        print(f"Graph image saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save graph image: {e}")
        return False

# --- Main Execution ---
def run_agentic_flow(user_input):
    """Run the agentic flow with user input"""
    try:
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        print("Input:", user_input)
        print("-" * 60)
        print("Processing through Agentic Flow...")
        print("-" * 60)
        
        output = graph.invoke(input_state)
        
        print("Final Output:")
        for message in output["messages"]:
            if hasattr(message, "content"):
                print(message.content)
            else:
                print(str(message))
                
        return output
    except Exception as e:
        print(f"Error running agentic flow: {e}")
        return None

if __name__ == "__main__":
    # Save graph visualization
    save_graph_image()
    
    # Example usage
    sample_input = """
    Artificial Intelligence is revolutionizing various industries. Machine learning algorithms can process vast amounts of data 
    to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural networks with multiple 
    layers to solve complex problems. AI applications include natural language processing, computer vision, robotics, and 
    autonomous vehicles. However, there are concerns about job displacement, privacy, and ethical implications of AI systems.
    """
    
    result = run_agentic_flow(sample_input.strip())
    
    if result:
        print("\n" + "="*60)
        print("Agentic Flow completed successfully!")
        print("="*60)
    else:
        print("\nFlow execution failed. Please check your setup.")