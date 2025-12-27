"""
Financial Analyst Team - Interactive CLI
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import logging
import sys
from langchain_core.messages import HumanMessage
from finance_analysts.graph import create_graph

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the financial analyst team."""
    
    print("=" * 60)
    print("🤖 FINANCIAL ANALYST TEAM")
    print("=" * 60)
    print("\nCapabilities:")
    print("  📊 Stock prices and analysis")
    print("  📰 Company news and fundamentals")
    print("  💼 Paper trading (buy/sell with fake money)")
    print("  📋 Comprehensive reports")
    print("\nExamples:")
    print("  • 'What's the price of AAPL?'")
    print("  • 'Compare Microsoft and Apple'")
    print("  • 'Buy 5 shares of NVDA'")
    print("  • 'Show my portfolio'")
    print("\nType 'quit' or 'exit' to quit")
    print("=" * 60)
    
    # Build the graph
    try:
        logger.info("\n🔨 Building agent graph...")
        graph = create_graph()
        logger.info("✅ Graph ready!\n")
    except Exception as e:
        logger.error(f"\n❌ Failed to build graph: {e}")
        logger.error("Make sure all dependencies are installed and .env is configured")
        sys.exit(1)
    
    # State for conversation
    state = {
        "messages": [],
        "results": {},
        "next_agents": []
    }
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            
            # Clear previous results for new query
            state["results"] = {}
            state["next_agents"] = []
            
            # Run the graph
            logger.info("\n🔄 Processing your request...")
            
            try:
                final_state = graph.invoke(state)
                
                # Update state with results
                state = final_state
                
                # Get the last AI message
                ai_response = None
                for msg in reversed(state["messages"]):
                    if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                        ai_response = msg.content
                        break
                
                if ai_response:
                    print(f"\n🤖 Assistant:\n{ai_response}")
                else:
                    logger.warning("\n⚠️ No response generated")
                    
            except Exception as e:
                logger.error(f"\n❌ Error: {e}")
                logger.info("Conversation state preserved, you can try again")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()