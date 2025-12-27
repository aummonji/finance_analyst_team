"""
Simple Streamlit UI for Financial Analyst Team
Run with: streamlit run ui_simple.py
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the script's directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Check if keys are loaded
with st.sidebar:
    st.header("🔑 API Keys Status")
    if os.getenv("ANTHROPIC_API_KEY"):
        st.success("✅ Anthropic API")
    else:
        st.error("❌ Anthropic API")
    
    if os.getenv("ALPACA_API_KEY"):
        st.success("✅ Alpaca API")
        # Show first few chars
        key = os.getenv("ALPACA_API_KEY")
        
    else:
        st.error("❌ Alpaca API")

from langchain_core.messages import HumanMessage
from finance_analysts.graph import create_graph

# Set page title
st.title("💬 Financial Analyst Chatbot")

# Add a button to force reload
if st.sidebar.button("🔄 Reload Graph"):
    if "graph" in st.session_state:
        del st.session_state.graph
    st.rerun()

# Initialize the graph (only once)
if "graph" not in st.session_state:
    with st.spinner("🔨 Building agent graph..."):
        # Clear any Python module cache
        import importlib
        import finance_analysts.agents
        import finance_analysts.graph
        importlib.reload(finance_analysts.agents)
        importlib.reload(finance_analysts.graph)
        
        st.session_state.graph = create_graph()
        st.sidebar.info("✅ Graph loaded")

# Initialize chat state (only once)
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "results": {},
        "next_agents": []
    }

# Initialize chat history for display (only once)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show all previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_input = st.chat_input("Ask me about stocks...")

# If user sent a message
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Save user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Add to state
    st.session_state.state["messages"].append(HumanMessage(content=user_input))
    st.session_state.state["results"] = {}
    st.session_state.state["next_agents"] = []
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Run the graph
                final_state = st.session_state.graph.invoke(st.session_state.state)
                st.session_state.state = final_state
                
                # Debug: Show what agents ran
                if final_state.get("results"):
                    with st.expander("🔍 Debug: Agents that ran"):
                        for agent, result in final_state["results"].items():
                            st.write(f"**{agent}:** {result[:200]}...")
                
                # Find the AI response
                response = None
                for msg in reversed(final_state["messages"]):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        response = msg.content
                        break
                
                # Show response
                if response:
                    st.write(response)
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    st.write("Sorry, no response generated.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Full error details"):
                    st.code(traceback.format_exc())