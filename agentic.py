#!/usr/bin/env python3
"""
Multi-LLM Agentic Flow using Ollama, LangChain, and LangGraph
This implementation creates a workflow where two LLM models communicate with each other.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

# Core imports
from langchain.llms import Ollama
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# LangGraph imports
from langgraph.graph import Graph, StateGraph, END

# State management - Use TypedDict for LangGraph compatibility
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """State shared between agents"""
    messages: List[BaseMessage]
    current_step: str
    model1_output: Optional[str]
    model2_output: Optional[str]
    final_result: Optional[str]
    iteration_count: int
    max_iterations: int

class MultiLLMAgent:
    """Multi-LLM Agent using Ollama models"""
    
    def __init__(self, model1_name: str = "llama3.2:3b", model2_name: str = "llama3.2:3b"):
        """
        Initialize the multi-LLM agent
        
        Args:
            model1_name: Name of the first Ollama model
            model2_name: Name of the second Ollama model
        """
        # Initialize Ollama models with proper base_url
        self.model1 = Ollama(
            model=model1_name, 
            temperature=0.7,
            base_url="http://localhost:11434"
        )
        self.model2 = Ollama(
            model=model2_name, 
            temperature=0.7,
            base_url="http://localhost:11434"
        )
        
        # Define system prompts for each model
        self.model1_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are Model 1 in a collaborative AI system. Your role is to provide initial analysis "
                "and insights on the given task. Be thorough but concise. You will pass your output "
                "to Model 2 for refinement and additional processing."
            ),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        self.model2_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are Model 2 in a collaborative AI system. You receive input from Model 1 "
                "and should build upon it, refine it, or provide complementary analysis. "
                "Model 1's output: {model1_output}"
            ),
            HumanMessagePromptTemplate.from_template("Original query: {input}")
        ])
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def start_node(state: AgentState) -> AgentState:
            """Initialize the workflow"""
            print("🚀 Starting multi-LLM workflow...")
            return state
        
        def model1_node(state: AgentState) -> AgentState:
            """Process with Model 1"""
            print("🤖 Model 1 processing...")
            
            # Get the latest human message
            human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if not human_messages:
                raise ValueError("No human message found in state")
            
            latest_input = human_messages[-1].content
            
            # Generate response from Model 1 using proper invoke method
            try:
                response = self.model1.invoke(latest_input)
                state["model1_output"] = response
                print(f"✅ Model 1 completed: {response[:100]}...")
            except Exception as e:
                print(f"❌ Model 1 error: {str(e)}")
                raise e
            
            return state
        
        def model2_node(state: AgentState) -> AgentState:
            """Process with Model 2"""
            print("🤖 Model 2 processing...")
            
            # Get the original input
            human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            latest_input = human_messages[-1].content
            
            # Create enhanced prompt for Model 2
            model2_input = f"""Original query: {latest_input}

Model 1's analysis: {state["model1_output"]}

Please build upon Model 1's analysis and provide additional insights or refinements."""
            
            # Generate response from Model 2
            try:
                response = self.model2.invoke(model2_input)
                state["model2_output"] = response
                state["final_result"] = self._combine_outputs(state["model1_output"], state["model2_output"])
                print(f"✅ Model 2 completed: {response[:100]}...")
            except Exception as e:
                print(f"❌ Model 2 error: {str(e)}")
                raise e
            
            return state
        
        def should_continue(state: AgentState) -> str:
            """Decide next step in workflow"""
            if state["current_step"] == "model1":
                return "model1"
            elif state["current_step"] == "model2":
                return "model2"
            else:
                return END
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("start", start_node)
        workflow.add_node("model1", model1_node)
        workflow.add_node("model2", model2_node)
        
        # Add edges - Create proper flow connections
        workflow.set_entry_point("start")
        workflow.add_edge("start", "model1")      # start → model1
        workflow.add_edge("model1", "model2")     # model1 → model2
        workflow.add_edge("model2", END)          # model2 → END
        
        return workflow.compile()
    
    def _combine_outputs(self, model1_output: str, model2_output: str) -> str:
        """Combine outputs from both models"""
        combined = f"""
=== COLLABORATIVE AI RESPONSE ===

🤖 Model 1 Analysis:
{model1_output}

🤖 Model 2 Refinement:
{model2_output}

=== SYNTHESIZED RESULT ===
Based on the collaborative analysis above, here's the comprehensive response:
{model2_output}
"""
        return combined
    
    async def process(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the multi-LLM workflow
        
        Args:
            query: The input query to process
            
        Returns:
            Dictionary containing the results from both models
        """
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "current_step": "start",
            "model1_output": None,
            "model2_output": None,
            "final_result": None,
            "iteration_count": 0,
            "max_iterations": 3
        }
        
        try:
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "query": query,
                "model1_output": final_state["model1_output"],
                "model2_output": final_state["model2_output"],
                "final_result": final_state["final_result"],
                "workflow_completed": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def process_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous version of process method"""
        return asyncio.run(self.process(query))
    
    def save_graph_diagram(self, filename: str = "multi_llm_workflow.png"):
        """Save the LangGraph workflow diagram"""
        try:
            from PIL import Image
            import io
            
            # Get the graph diagram
            diagram = self.graph.get_graph()
            
            # Save as PNG
            img_data = diagram.draw_mermaid_png()
            if img_data:
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"📊 Graph diagram saved as {filename}")
            else:
                print("❌ Could not generate diagram")
                
        except ImportError:
            print("❌ PIL (Pillow) not installed. Install with: pip install pillow")
        except Exception as e:
            print(f"❌ Error saving diagram: {str(e)}")
    
    def print_graph_structure(self):
        """Print the graph structure in text format"""
        print("\n📊 LangGraph Workflow Structure:")
        print("=" * 40)
        print("START")
        print("  ↓")
        print("MODEL1 (llama3.2:3b)")
        print("  ↓")
        print("MODEL2 (llama3.2:3b)")
        print("  ↓")
        print("END")
        print("=" * 40)
        
        # Print node details
        print("\n🔍 Node Details:")
        print("• START: Initialize workflow")
        print("• MODEL1: First LLM processes original query")
        print("• MODEL2: Second LLM refines using MODEL1 output")
        print("• END: Return combined results")
        
        print("\n🔗 Edge Connections:")
        print("• START → MODEL1")
        print("• MODEL1 → MODEL2") 
        print("• MODEL2 → END")
        
        try:
            graph = self.graph.get_graph()
            print(f"\n🔢 Graph Statistics:")
            print(f"• Nodes: {len(graph.nodes)}")
            print(f"• Edges: {len(graph.edges)}")
            print("• Flow: Linear sequence (no conditionals)")
        except Exception as e:
            print(f"❌ Could not get graph stats: {str(e)}")

# Enhanced version with memory and conversation history
class ConversationalMultiLLMAgent(MultiLLMAgent):
    """Multi-LLM Agent with conversation memory"""
    
    def __init__(self, model1_name: str = "llama3.2:3b", model2_name: str = "llama3.2:3b"):
        super().__init__(model1_name, model2_name)
        self.conversation_history: List[Dict[str, Any]] = []
    
    def add_to_history(self, query: str, result: Dict[str, Any]):
        """Add interaction to conversation history"""
        self.conversation_history.append({
            "timestamp": asyncio.get_event_loop().time(),
            "query": query,
            "result": result
        })
    
    async def process_with_memory(self, query: str) -> Dict[str, Any]:
        """Process query with conversation memory"""
        # Add context from previous conversations if available
        context = ""
        if self.conversation_history:
            context = "Previous conversation context:\n"
            for i, conv in enumerate(self.conversation_history[-2:]):  # Last 2 conversations
                context += f"Q{i+1}: {conv['query']}\nA{i+1}: {conv['result'].get('model2_output', '')[:200]}...\n\n"
        
        enhanced_query = f"{context}Current query: {query}" if context else query
        result = await self.process(enhanced_query)
        
        # Add to history
        self.add_to_history(query, result)
        
        return result

# Example usage and testing functions
def test_multi_llm_agent():
    """Test the multi-LLM agent"""
    print("🧪 Testing Multi-LLM Agent...")
    
    # Initialize agent with llama3.2:3b
    agent = MultiLLMAgent(model1_name="llama3.2:3b", model2_name="llama3.2:3b")
    
    # Print graph structure
    agent.print_graph_structure()
    
    # Try to save graph diagram
    agent.save_graph_diagram()
    
    # Test queries
    test_queries = [
        "Explain the concept of quantum computing",
        "What are the benefits and challenges of renewable energy?",
        "How can AI be used in healthcare?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = agent.process_sync(query)
            
            if result["success"]:
                print("✅ Success!")
                print(f"🎯 Final Result:\n{result['final_result']}")
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            print("💡 Make sure Ollama is running and llama3.2:3b is available")

def test_conversational_agent():
    """Test the conversational multi-LLM agent"""
    print("\n🗣️ Testing Conversational Multi-LLM Agent...")
    
    agent = ConversationalMultiLLMAgent()
    
    conversation_flow = [
        "What is machine learning?",
        "How does it differ from traditional programming?",
        "Can you give me a practical example?"
    ]
    
    for query in conversation_flow:
        print(f"\n👤 User: {query}")
        result = asyncio.run(agent.process_with_memory(query))
        
        if result["success"]:
            print(f"🤖 AI: {result['model2_output'][:300]}...")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    # Make sure you have Ollama running with the required models
    print("🚀 Multi-LLM Agentic Flow Demo")
    print("=" * 50)
    print("Prerequisites:")
    print("1. Ollama should be running (ollama serve)")
    print("2. Model should be pulled (ollama pull llama3.2:3b)")
    print("3. Required packages: langchain langgraph ollama")
    print("=" * 50)
    
    try:
        # Run tests
        test_multi_llm_agent()
        test_conversational_agent()
        
    except Exception as e:
        print(f"❌ Setup Error: {str(e)}")
        print("Please ensure Ollama is running and models are available.")