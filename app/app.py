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
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Create a default config if file doesn't exist
        default_config = {
            "app/summarize.txt": {
                "model": "llama3.2:3b",
                "temperature": 0.7
            },
            "app/summarize_agentic.txt": {
                "model": "llama3.2:3b", 
                "temperature": 0.3
            }
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config file: {config_path}")
        return default_config
    
    with open(config_path, "r") as f:
        return json.load(f)

def get_llm(config: dict, file_key: str):
    """Get LLM instance from config"""
    settings = config.get(file_key, {"model": "llama3.2:3b", "temperature": 0.7})
    return ChatOllama(
        model=settings["model"],
        temperature=settings.get("temperature", 0.7)
    )

def load_prompt_template(template_path):
    """Load prompt template with fallback"""
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            return f.read()
    else:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        # Default prompt template
        default_prompt = """You are a helpful AI assistant specialized in providing detailed information.

Query: {input}

Please provide a comprehensive and accurate response. If you need additional processing or analysis, use the available tools."""
        
        # Save default prompt to file
        with open(template_path, "w") as f:
            f.write(default_prompt)
        print(f"Created default prompt template: {template_path}")
        
        return default_prompt

# Load configuration and models
try:
    config = load_config("app/gemma3_year.json")
    prompt_template = load_prompt_template("app/summarize_agentic.txt")
    llm_summarize = get_llm(config, "app/summarize.txt")
    llm_summarize_agentic = get_llm(config, "app/summarize_agentic.txt")
    print("✅ Configuration and models loaded successfully!")
except Exception as e:
    print(f"Warning: Configuration loading failed: {e}")
    # Fallback configuration
    llm_summarize_agentic = ChatOllama(model="llama3.2:3b", temperature=0.3)
    llm_summarize = ChatOllama(model="llama3.2:3b", temperature=0.7)
    prompt_template = """You are a helpful AI assistant specialized in providing detailed information.

Query: {input}

Please provide a comprehensive and accurate response. If you need additional processing or analysis, use the available tools."""
    print("✅ Using fallback configuration!")

# --- Tools ---

@tool
def search_with_llm(query: str) -> str:
    """Search and analyze using secondary LLM - converts detailed content to structured points"""
    try:
        # This is your second model that processes the query
        analysis_prompt = f"""You are a specialized analysis assistant. Take the following query and provide a structured, point-by-point analysis.

Query: {query}

Provide your response in clear bullet points or structured format:"""
        
        response = llm_summarize.invoke(analysis_prompt)
        
        # Handle different response types
        if isinstance(response, dict):
            result = response.get("content", str(response))
        elif hasattr(response, "content"):
            result = response.content
        else:
            result = str(response)
            
        return f"📊 Structured Analysis:\n{result}"
        
    except Exception as e:
        return f"Error in search_with_llm: {str(e)}"

@tool
def human_assistance(query: str) -> str:
    """Request human assistance (simplified version)"""
    try:
        print(f"\n🤖 AI Assistant is requesting help with: {query}")
        print("=" * 60)
        print("💡 In a real scenario, this would pause for human input.")
        print("💡 For this demo, providing automated response...")
        
        # Simulated human response for demo purposes
        demo_response = f"Human guidance: Focus on practical examples and step-by-step instructions for '{query}'. Make sure to include installation steps and common troubleshooting tips."
        
        return f"👤 Human Input: {demo_response}"
        
    except Exception as e:
        return f"Human assistance not available: {str(e)}"

# Available tools
tools = [search_with_llm, human_assistance]

# --- Bind tools to the agentic LLM ---
chatbot = llm_summarize_agentic.bind_tools(tools)

# --- LangGraph Setup ---

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state):
    """Main chatbot node function"""
    try:
        response = chatbot.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"Error in chatbot_node: {str(e)}"
        print(error_msg)
        return {"messages": [error_msg]}

# Create graph
graph_builder = StateGraph(State)

# Add nodes
print("🔧 Building LangGraph...")
graph_builder.add_node("chatbot", chatbot_node)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges following the flow: START -> chatbot -> (tools OR END) -> chatbot
graph_builder.add_edge(START, "chatbot")

# Conditional edges from chatbot
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # Map: if tools are called -> "tools", otherwise -> END
)

# After tools execution, go back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Compile graph
graph = graph_builder.compile()
print("✅ LangGraph compiled successfully!")

# --- Graph Visualization ---
def save_graph_image():
    """Save graph visualization"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        image_data = graph.get_graph().draw_mermaid_png()
        output_path = "output/graph.png"

        with open(output_path, "wb") as f:
            f.write(image_data)

        print(f"📊 Graph image saved to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save graph image: {e}")
        return False

# --- Main Execution ---
def run_graph_example(custom_query=None):
    """Run the graph with example query"""
    try:
        if custom_query:
            user_query = custom_query
        else:
            user_query = "What is Ollama and how can I use it for local LLMs? Provide detailed setup instructions."
            
        input_state = {"messages": [HumanMessage(content=user_query)]}
        
        print("\n" + "="*80)
        print("🚀 STARTING LANGGRAPH EXECUTION")
        print("="*80)
        print(f"📝 Query: {user_query}")
        print("-" * 80)
        
        # Execute the graph and stream results
        print("🔄 Processing through LangGraph flow...")
        
        final_output = graph.invoke(input_state)
        
        print("\n" + "="*80)
        print("📋 EXECUTION RESULTS")
        print("="*80)
        
        for i, message in enumerate(final_output["messages"], 1):
            print(f"\n--- Message {i}: {message.__class__.__name__} ---")
            if hasattr(message, "content"):
                content = message.content
                if content.strip():  # Only print non-empty content
                    print(content)
            elif hasattr(message, "tool_calls") and message.tool_calls:
                print("🔧 Tool calls:", [tool_call["name"] for tool_call in message.tool_calls])
            else:
                print(f"Message type: {type(message)}")
                
        return final_output
        
    except Exception as e:
        print(f"❌ Error running graph: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_interactive_mode():
    """Run in interactive mode"""
    print("\n🤖 Interactive LangGraph Mode")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n👤 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                print("Please enter a query.")
                continue
                
            run_graph_example(user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 LangGraph with Ollama - Complete Implementation")
    print("=" * 60)
    
    # Save graph visualization
    print("\n📊 Generating graph visualization...")
    save_graph_image()
    
    # Run example
    print("\n🧪 Running test example...")
    result = run_graph_example()
    
    if result:
        print("\n✅ Test execution completed successfully!")
        
        # Ask if user wants interactive mode
        while True:
            choice = input("\n🤔 Would you like to try interactive mode? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                run_interactive_mode()
                break
            elif choice in ['n', 'no']:
                print("👋 Thanks for testing!")
                break
            else:
                print("Please enter 'y' or 'n'")
                
    else:
        print("\n❌ Test execution failed.")
        print("💡 Make sure you have:")
        print("   1. Ollama installed and running")
        print("   2. llama3.2:3b model pulled (ollama pull llama3.2:3b)")
        print("   3. All required Python packages installed")