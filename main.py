from openai import OpenAI
import asyncio
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
import os
from dotenv import load_dotenv


load_dotenv()

set_default_openai_key(os.getenv('OPENAI_API_KEY'))

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'),)


def upload_file(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        attach_response = client.vector_stores.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )

        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}


def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}
    
    
# vector_store_id = create_vector_store("ACME Shop Product Knowledge Base")
# upload_file("voice_agents_knowledge/acme_product_catalogue.pdf", vector_store_id["id"])


# --- Agent: Search Agent ---
search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],

)


# # --- Agent: Knowledge Agent ---
# knowledge_agent = Agent(
#     name="KnowledgeAgent",
#     instructions=(
#         "You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool."
#     ),
#     tools=[FileSearchTool(
#             max_num_results=3,
#             vector_store_ids=["VECTOR_STORE_ID"],
#         ),],

# )

# --- Tool 1: Fetch account information (dummy) ---
@function_tool
def get_account_info(user_id: str) -> dict:
    """Return dummy account info for a given user."""
    return {
        "user_id": user_id,
        "name": "Bugs Bunny",
        "account_balance": "£72.50",
        "membership_status": "Gold Executive"
    }

# --- Agent: Account Agent ---
account_agent = Agent(
    name="AccountAgent",
    instructions=(
        "You provide account information based on a user ID using the get_account_info tool."
    ),
    tools=[get_account_info],

)


# --- Agent: Triage Agent ---
triage_agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for Acme Shop. Welcome the user and ask how you can help.
Based on the user's intent, route to:
- AccountAgent for account-related queries
- SearchAgent for anything requiring real-time web search
"""),
    handoffs=[account_agent, search_agent],

)


from agents import Runner, trace

async def test_queries():
    examples = [
        "What's my ACME account balance doc? My user ID is 1234567890", # Account Agent test
        "Ooh i've got money to spend! How big is the input and how fast is the output of the dynamite dispenser?", # Knowledge Agent test
        "Hmmm, what about duck hunting gear - what's trending right now?", # Search Agent test

    ]
    with trace("ACME App Assistant"):
        for query in examples:
            result = await Runner.run(triage_agent, query)
            print(f"User: {query}")
            print(result.final_output)
            print("---")
# Run the tests
asyncio.run(test_queries())

