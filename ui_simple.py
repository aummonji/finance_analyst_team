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

# =============================================================================
# SIDEBAR - Shows on the left side of the page
# =============================================================================
# st.sidebar.X puts elements in the sidebar instead of main area
with st.sidebar:
    st.header("🔑 API Keys Status")
    
    # Check if environment variables are loaded
    if os.getenv("ANTHROPIC_API_KEY"):
        st.success("✅ Anthropic API")  # Green box
    else:
        st.error("❌ Anthropic API")    # Red box
    
    if os.getenv("ALPACA_API_KEY"):
        st.success("✅ Alpaca API")
    else:
        st.error("❌ Alpaca API")

from langchain_core.messages import HumanMessage
from finance_analysts.graph import create_graph

# =============================================================================
# PAGE SETUP
# =============================================================================
# st.title() - Big header at top of page
st.title("💬 Your Finance Analysts")

# Reload button - for development, forces graph to rebuild when you change agent code
if st.sidebar.button("🔄 Reload Graph"):
    if "graph" in st.session_state:
        del st.session_state.graph
    st.rerun()  # Reruns the entire script

# =============================================================================
# SESSION STATE - Persists data between reruns
# =============================================================================
# Streamlit reruns the ENTIRE script every time user interacts (clicks, types, etc)
# st.session_state is a dict that persists between reruns
# Without it, variables reset to initial values on every interaction

# Initialize the graph (only once, not on every rerun)
if "graph" not in st.session_state:
    # st.spinner() - Shows loading animation while code inside runs
    with st.spinner("🔨 Building agent graph..."):
        st.session_state.graph = create_graph()

# Initialize LangGraph state (messages, results, next_agents)
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "results": {},
        "next_agents": []
    }

# Initialize chat history for UI display
# (separate from LangGraph state - this is just for showing messages)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================
# Loop through previous messages and display them
for message in st.session_state.chat_history:
    # st.chat_message() - Creates a chat bubble with avatar
    # role="user" shows person icon, role="assistant" shows bot icon
    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "assistant":
            content = content.replace("$", "\\$")  # Fix LaTeX rendering
        st.write(content)

# =============================================================================
# CHAT INPUT
# =============================================================================
# st.chat_input() - Creates the text input box at bottom of page
# Returns None if empty, returns text when user presses Enter
user_input = st.chat_input("Ask me about stocks...")

# =============================================================================
# HANDLE USER INPUT
# =============================================================================
if user_input:
    # Show user's message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Save to chat history (for display on next rerun)
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Add to LangGraph state
    st.session_state.state["messages"].append(HumanMessage(content=user_input))
    st.session_state.state["results"] = {}       # Clear previous results
    st.session_state.state["next_agents"] = []   # Clear previous routing
    
    # Show assistant response
    with st.chat_message("assistant"):
        # st.spinner() - Shows "Thinking..." while graph runs
        with st.spinner("Thinking..."):
            try:
                # Run the LangGraph
                final_state = st.session_state.graph.invoke(st.session_state.state)
                st.session_state.state = final_state
                
                # Show which agents ran
                # st.caption() - Small gray text, good for metadata
                if final_state.get("results"):
                    agents_ran = list(final_state["results"].keys())
                    st.caption(f"🎯 Routed: {' → '.join(agents_ran)}")
                
                # Find the AI response (last AIMessage in state)
                response = None
                for msg in reversed(final_state["messages"]):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        response = msg.content
                        break
                
                # Display response
                if response:
                    response = response.replace("$", "\\$")  # Fix LaTeX rendering
                    st.write(response)
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    st.write("Sorry, no response generated.")
                    
            except Exception as e:
                # st.error() - Red error box
                st.error(f"❌ Error: {str(e)}")
                import traceback
                # st.expander() - Collapsible section
                with st.expander("Full error details"):
                    # st.code() - Monospace code block
                    st.code(traceback.format_exc())