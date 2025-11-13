import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

TRAVEL_AGENT_PROMPT = """You are an expert AI Travel Planning Assistant with access to real-time travel data through the BrightData MCP tool.

Your core responsibilities:
1. **Flight Search & Comparison**: Find and compare flights across airlines, considering price, duration, stops, and departure times
2. **Hotel Research**: Search for accommodations matching user preferences (budget, location, amenities, ratings)
3. **Itinerary Planning**: Create day-by-day travel plans with activities, attractions, and local recommendations
4. **Travel Intelligence**: Provide visa requirements, weather forecasts, local customs, and safety information
5. **Budget Optimization**: Help users find the best value options while respecting their budget constraints

When handling requests:
- Always confirm dates, locations, and passenger/guest counts before searching
- Present flight options sorted by best value (balance of price, duration, and convenience)
- For hotels, highlight key features like location proximity, amenities, and guest ratings
- Provide context: travel times, time zone changes, layover durations
- Suggest alternatives if initial search yields limited results
- Include practical tips: best booking times, seasonal considerations, local insights

Current date context: {current_date}

Communication style:
- Be conversational yet professional
- Ask clarifying questions when needed
- Present information in clear, scannable formats
- Highlight important details (prices, times, restrictions)
- Proactively suggest relevant add-ons (transfers, insurance, tours)

Always prioritize accuracy and user safety in your recommendations."""


class TravelPlanningAgent:
    def __init__(self):
        self.client = None
        self.agent = None
        self.current_date = datetime.now().strftime("%B %d, %Y")
        
    async def initialize(self):
        try:
            self.client = MultiServerMCPClient(
                {
                    "bright_data": {
                        "command": "npx",
                        "args": ["@brightdata/mcp"],
                        "env": {
                            "API_TOKEN": os.getenv("BRIGHT_DATA_API")
                        },
                        "transport": "stdio",
                    }
                }
            )
            
            tools = await self.client.get_tools()
            print(f"Loaded {len(tools)} tools from BrightData MCP")
       
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0.3
            )
            
            self.agent = create_react_agent(llm, tools)
            self.system_message = TRAVEL_AGENT_PROMPT.format(current_date=self.current_date)
            
            print("Travel Planning Agent initialized successfully\n")
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise
    
    async def plan_trip(self, user_request: str):
        try:
            print(f"Processing request: {user_request}\n")
            print("=" * 80)
            
            from langchain_core.messages import SystemMessage
            
            messages = [
                SystemMessage(content=self.system_message),
                HumanMessage(content=user_request)
            ]
            
            response = await self.agent.ainvoke({
                "messages": messages
            })
            
            final_message = response["messages"][-1].content
            
            print("\n" + "=" * 80)
            print("TRAVEL PLAN RESPONSE:")
            print("=" * 80)
            print(final_message)
            print("=" * 80 + "\n")
            
            return final_message
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(f"{error_msg}")
            return error_msg
    
    async def interactive_mode(self):
        print("\n" + "=" * 80)
        print("TRAVEL PLANNING ASSISTANT")
        print("=" * 80)
        print("I can help you with:")
        print("  • Flight searches and comparisons")
        print("  • Hotel recommendations")
        print("  • Complete trip itineraries")
        print("  • Travel tips and local insights")
        print("\nType 'quit' or 'exit' to end the session")
        print("=" * 80 + "\n")
        
        from langchain_core.messages import SystemMessage
        
        # Initialize with system message
        conversation_history = [SystemMessage(content=self.system_message)]
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n Safe travels! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Add to conversation history
                conversation_history.append(HumanMessage(content=user_input))
                
                # Get response
                response = await self.agent.ainvoke({
                    "messages": conversation_history
                })
                
                # Extract and display response
                assistant_message = response["messages"][-1]
                conversation_history.append(assistant_message)
                
                print(f"\nAssistant: {assistant_message.content}\n")
                print("-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Safe travels!")
                break
            except Exception as e:
                print(f"\n✗ Error: {str(e)}\n")
    
    async def cleanup(self):
        if self.client:
            # Add cleanup logic if needed
            pass


async def main():

    agent = TravelPlanningAgent()
    
    try:
        await agent.initialize()
        
        # example_requests = [
        #     #"Tell me available flights from Dubai to Colombo on December 1, 2025. I'm looking for the best balance of price and convenience.",
        #     #"I need a 5-day trip plan to Sri Lanka in March 2025, budget $2000 including flights from Dubai",
        #     "Find me mid-range hotels in Paris near the Eiffel Tower for 3 nights in April"
        # ]
        
        # await agent.plan_trip(example_requests[0])
        await agent.interactive_mode()

    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())