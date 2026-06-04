# =================================================================
# Challenge 3 - Memory Agent
# Strands SDK + Ollama llama3.2:3b + Mem0 + FAISS
# =================================================================

from mem0 import Memory
from strands import Agent
from strands.models.ollama import OllamaModel

# ------------------------------------------------------------------
# MEM0 CONFIG
# ------------------------------------------------------------------

MEM0_CONFIG = {
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": "challenge3_memories",
            "path": "./memory_store",
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 768,
        },
    },
}

# ------------------------------------------------------------------
# INIT
# ------------------------------------------------------------------

print("Loading memory system ...")

memory = Memory.from_config(MEM0_CONFIG)

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)

USER_ID = "sujith"


def build_agent(recalled_memories):
    system_prompt = (
        "You are a helpful AI assistant with persistent memory.\n\n"
        "Memories from previous conversations:\n"
        + recalled_memories
        + "\n\nUse these memories to personalize every reply."
    )

    return Agent(
        model=ollama_model,
        system_prompt=system_prompt,
    )


# ------------------------------------------------------------------
# MEMORY HELPERS
# ------------------------------------------------------------------

def save_to_memory(user_message, agent_reply):
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": agent_reply},
    ]

    try:
        memory.add(
            messages,
            user_id=USER_ID
        )
        print("[Memory Saved]")

    except Exception as e:
        print("[Memory Save Error]", e)


def recall_memories(query):
    try:
        results = memory.search(
            query=query,
            filters={"user_id": USER_ID},
            limit=5,
        )

        entries = (
            results.get("results", results)
            if isinstance(results, dict)
            else results
        )

        if not entries:
            return "No memories stored yet."

        return "\n".join(
            f"{i}. {e.get('memory', str(e))}"
            for i, e in enumerate(entries, 1)
        )

    except Exception as e:
        print("[Memory Search Error]", e)
        return "No memories stored yet."


def show_all_memories():
    try:
        all_mem = memory.get_all(
            filters={"user_id": USER_ID}
        )

        entries = (
            all_mem.get("results", all_mem)
            if isinstance(all_mem, dict)
            else all_mem
        )

        if not entries:
            print("  (no memories stored yet)")
            return

        for i, e in enumerate(entries, 1):
            print(f"  {i}. {e.get('memory', str(e))}")

    except Exception as e:
        print("[Memory Read Error]", e)


# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------

def main():
    print("=" * 58)
    print("  Challenge 3 - Memory Agent")
    print("  Model  : llama3.2:3b (Ollama)")
    print("  Memory : Mem0 + FAISS")
    print("  Type 'memories' to see stored facts")
    print("  Type 'quit' to exit")
    print("=" * 58)
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "memories":
            print("\nStored Memories:")
            show_all_memories()
            print()
            continue

        recalled = recall_memories(user_input)

        agent = build_agent(recalled)

        print("\nAgent: ", end="", flush=True)

        result = agent(user_input)
        agent_reply = str(result)

        print(agent_reply)

        save_to_memory(user_input, agent_reply)


if __name__ == "__main__":
    main()