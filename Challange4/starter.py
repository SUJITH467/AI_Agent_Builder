# =================================================================
# Challenge 4 – Full Agent
# Strands SDK + Ollama llama3.2:3b + Mem0 + FAISS
#
# This is the complete agent combining everything from
# Challenge 2 (tools) and Challenge 3 (persistent memory).
#
# Every user turn follows this pipeline:
#   1. Recall relevant memories from FAISS (vector search)
#   2. Build a Strands Agent with those memories in its prompt
#   3. Agent decides whether to call a tool or answer directly
#   4. Reply is streamed to the terminal
#   5. The turn is saved back into Mem0 / FAISS for future recall
# =================================================================


# -----------------------------------------------------------------
# SECTION 1 – IMPORTS
# -----------------------------------------------------------------

from datetime import date                      # used by age_calculator

from mem0 import Memory                        # persistent memory layer
from strands import Agent, tool                # agent + tool decorator
from strands.models.ollama import OllamaModel  # local Ollama adapter


# -----------------------------------------------------------------
# SECTION 2 – TOOLS
#
# Each tool is a plain Python function tagged with @tool.
# The decorator reads the name, docstring, and type hints to build
# the JSON schema the LLM needs to decide when/how to call it.
# Tools are passed to the Agent at initialisation time.
# -----------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this tool for any arithmetic, algebra, or numeric calculation.

    Args:
        expression: A valid Python math expression such as '2 + 2',
                    '10 * (3 + 4)', or 'sqrt(144)'.
    """
    import math
    try:
        # Only math functions and numbers are exposed to eval —
        # __builtins__ is blocked so no dangerous code can run.
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_weather(city: str) -> str:
    """
    Return a weather report for a given city.
    Use this tool whenever the user asks about weather, temperature,
    or forecast for any location.

    Args:
        city: Name of the city, e.g. 'London', 'Tokyo', 'New York'.
    """
    # Mock data – replace the dict lookup with a real API call
    # (e.g. OpenWeatherMap) when you're ready to go live.
    mock = {
        "london":   {"temp": "15°C", "condition": "Cloudy",       "humidity": "78%"},
        "new york": {"temp": "22°C", "condition": "Sunny",        "humidity": "55%"},
        "tokyo":    {"temp": "28°C", "condition": "Humid",        "humidity": "82%"},
        "sydney":   {"temp": "19°C", "condition": "Windy",        "humidity": "65%"},
        "paris":    {"temp": "17°C", "condition": "Rainy",        "humidity": "80%"},
        "mumbai":   {"temp": "32°C", "condition": "Hot & Humid",  "humidity": "88%"},
        "dubai":    {"temp": "38°C", "condition": "Sunny & Hot",  "humidity": "45%"},
    }
    data = mock.get(city.lower().strip())
    if data:
        return (
            f"Weather in {city.title()}: {data['condition']}, "
            f"Temp: {data['temp']}, Humidity: {data['humidity']}"
        )
    return f"Weather in {city.title()}: Partly Cloudy, Temp: 20°C, Humidity: 60%"


@tool
def age_calculator(birth_year: int, birth_month: int, birth_day: int) -> str:
    """
    Calculate a person's exact age from their date of birth.
    Use this tool when the user asks how old someone is or wants
    to know the age from a birthdate.

    Args:
        birth_year:  Four-digit year, e.g. 1990.
        birth_month: Month as integer (1 = January … 12 = December).
        birth_day:   Day as integer (1-31).
    """
    try:
        today     = date.today()
        birthdate = date(birth_year, birth_month, birth_day)

        if birthdate > today:
            return "That birth date is in the future – please check the values."

        age = today.year - birthdate.year
        # Subtract 1 if the birthday hasn't happened yet this year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            age -= 1

        next_bday = birthdate.replace(year=today.year)
        if next_bday <= today:
            next_bday = next_bday.replace(year=today.year + 1)
        days_left = (next_bday - today).days

        return (
            f"Age: {age} years old. "
            f"Next birthday in {days_left} day(s) "
            f"({next_bday.strftime('%B %d, %Y')})."
        )
    except ValueError as e:
        return f"Invalid date: {e}"


# -----------------------------------------------------------------
# SECTION 3 – MEM0 + FAISS CONFIGURATION
#
# Three local components – no API keys, no cloud:
#
#   vector_store → FAISS          : saves memory vectors to disk
#   llm          → Ollama         : Mem0 uses this to extract facts
#   embedder     → nomic-embed-text : turns text into 768-dim vectors
#
# The ./memory_store/ folder is created automatically on first run
# and reloaded on every subsequent run, giving true persistence.
# -----------------------------------------------------------------

MEM0_CONFIG = {
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": "challenge4_memories",
            "path": "./memory_store",        # persists between runs
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0,                # deterministic extraction
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",     # pull with: ollama pull nomic-embed-text
            "ollama_base_url": "http://localhost:11434",
        },
    },
}


# -----------------------------------------------------------------
# SECTION 4 – INITIALISE MEM0 AND OLLAMA MODEL
# -----------------------------------------------------------------

print("Initialising memory system …")
memory = Memory.from_config(MEM0_CONFIG)

# Shared Ollama model instance – used by every agent we build
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,   # low = precise tool calls + factual recall
)

# Every user turn gets a fresh Agent whose system prompt contains
# the recalled memories for that specific question.
TOOLS = [calculator, get_weather, age_calculator]


# -----------------------------------------------------------------
# SECTION 5 – MEMORY HELPERS
# -----------------------------------------------------------------

USER_ID = "sujith"


def recall_memories(query: str) -> str:
    try:
        results = memory.search(
            query=query,
            top_k=5,
            filters={"user_id": USER_ID}
        )

        if not results:
            return "No memories stored yet."

        memories = []

        for i, item in enumerate(results, 1):
            if isinstance(item, dict):
                memories.append(
                    f"{i}. {item.get('memory', str(item))}"
                )
            else:
                memories.append(f"{i}. {item}")

        return "\n".join(memories)

    except Exception:
        import traceback
        print("\n===== MEMORY SEARCH TRACEBACK =====")
        traceback.print_exc()
        print("===================================\n")
        return "No memories stored yet."


def save_to_memory(user_msg: str, agent_reply: str) -> None:
    try:
        print("Attempting memory save...")

        result = memory.add(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": agent_reply},
            ],
            user_id=USER_ID,
        )

        print("Memory add result:", result)

    except Exception:
        import traceback
        print("\n===== MEMORY SAVE TRACEBACK =====")
        traceback.print_exc()
        print("=================================\n")


def show_all_memories() -> None:
    try:
        memories = memory.get_all(
            filters={"user_id": USER_ID}
        )

        if not memories:
            print("  (no memories stored yet)")
            return

        for i, item in enumerate(memories, 1):
            if isinstance(item, dict):
                print(
                    f"  {i}. {item.get('memory', str(item))}"
                )
            else:
                print(f"  {i}. {item}")

    except Exception:
        import traceback
        print("\n===== MEMORY FETCH TRACEBACK =====")
        traceback.print_exc()
        print("==================================\n")
# -----------------------------------------------------------------
# SECTION 6 – AGENT BUILDER
#
# We rebuild the Agent on every turn so the system prompt always
# contains the freshest memories relevant to the current question.
# The tools list is constant – all three tools are always available.
# -----------------------------------------------------------------

def build_agent(recalled: str) -> Agent:
    """Build a Strands Agent pre-loaded with recalled memories."""
    system_prompt = (
        "You are a helpful AI assistant with tools and persistent memory.\n\n"
        "Available tools (use them when appropriate):\n"
        "  • calculator      – any arithmetic or math\n"
        "  • get_weather     – weather for a city\n"
        "  • age_calculator  – calculate age from a birth date\n\n"
        "You also have memories from previous conversations.\n"
        "ALWAYS use these to personalise your replies:\n\n"
        f"=== MEMORIES ===\n{recalled}\n================\n\n"
        "If no memories exist yet, answer normally and learn from "
        "this conversation."
    )
    return Agent(
        model=ollama_model,
        system_prompt=system_prompt,
        tools=TOOLS,
    )


# -----------------------------------------------------------------
# SECTION 7 – INTERACTIVE CHAT LOOP
#
# Each turn:
#   a) Read user input
#   b) recall_memories()  → top-5 FAISS hits for this message
#   c) build_agent()      → inject those memories into system prompt
#   d) agent(user_input)  → LLM replies (calls tools if needed)
#   e) save_to_memory()   → new facts written back to FAISS
# -----------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Challenge 4 – Full Agent")
    print("  Model   : llama3.2:3b  (Ollama, local)")
    print("  Tools   : calculator | get_weather | age_calculator")
    print("  Memory  : Mem0 + FAISS  (./memory_store/)")
    print("  Commands: 'memories' | 'tools' | 'quit'")
    print("=" * 60)
    print()
    print("Try asking:")
    print("  • My name is Sujith J and I live in Chennai")
    print("  • What is 128 * 256?")
    print("  • What's the weather in Mumbai?")
    print("  • How old is someone born on June 10, 1995?")
    print("  • What is my name?  ← works even after restarting")
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # ── exit ──────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! Memories saved to ./memory_store/")
            break

        # ── list all stored memories ──────────────────────────
        if user_input.lower() == "memories":
            print("\n── Stored memories ──")
            show_all_memories()
            print()
            continue

        # ── list loaded tools ─────────────────────────────────
        if user_input.lower() == "tools":
            # Build a temporary agent just to read tool_names
            agent = build_agent("")
            print("Loaded tools:", agent.tool_names)
            print()
            continue

        # ── normal chat turn ──────────────────────────────────

        # Step 1: find relevant memories for this message
        recalled = recall_memories(user_input)

        # Step 2: create agent with those memories in its prompt
        agent = build_agent(recalled)

        # Step 3: run the agent – it streams the reply to stdout
        #         and returns a result object we can convert to str
        print("\nAgent: ", end="", flush=True)
        result = agent(user_input)
        agent_reply = str(result)
        print()  # blank line between turns

        # Step 4: persist this turn so facts can be recalled later
        save_to_memory(user_input, agent_reply)


if __name__ == "__main__":
    main()
