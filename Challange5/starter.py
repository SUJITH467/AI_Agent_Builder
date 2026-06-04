# =================================================================
# Challenge 5 – Innovate MCP Chatbot
# Strands SDK + Ollama llama3.2:3b + Local MCP Server
#
# Architecture:
#
#   starter.py  (this file)
#       │
#       │  launches as subprocess (stdio transport)
#       ▼
#   mcp_server.py  ← the MCP server (4 tools)
#
# The Strands Agent connects to the MCP server, discovers its
# tools automatically, and uses them to answer questions.
#
# MCP (Model Context Protocol) is a standard that lets you expose
# tools as a separate service.  The agent doesn't need to know
# how the tools are implemented — it just calls them by name.
# =================================================================


# -----------------------------------------------------------------
# SECTION 1 – IMPORTS
# -----------------------------------------------------------------

import sys                                     # to get current Python path
from strands import Agent                      # Strands agent core
from strands.models.ollama import OllamaModel  # local Ollama adapter
from strands.tools.mcp import MCPClient        # Strands MCP client
from mcp import StdioServerParameters          # stdio transport config


# -----------------------------------------------------------------
# SECTION 2 – MCP SERVER CONNECTION
#
# StdioServerParameters tells Strands how to launch the MCP server:
#   command  – the executable (Python interpreter)
#   args     – the script to run (our mcp_server.py)
#
# Strands starts mcp_server.py as a child process and talks to it
# over stdin/stdout using the MCP JSON-RPC protocol.
# Everything is local — no network ports, no Docker.
# -----------------------------------------------------------------

from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

mcp_server_params = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"],
)

mcp_client = MCPClient(
    lambda: stdio_client(mcp_server_params)
)

# MCPClient manages the subprocess lifecycle.
# Used as a context manager (with block) so the server process is
# started on enter and cleanly terminated on exit.



# -----------------------------------------------------------------
# SECTION 3 – OLLAMA MODEL
#
# Same OllamaModel setup as previous challenges.
# temperature=0.3 keeps tool selection precise.
# -----------------------------------------------------------------

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,
)


# -----------------------------------------------------------------
# SECTION 4 – AGENT BUILDER
#
# mcp_client.list_tools_sync() asks the MCP server which tools
# it exposes and returns them as Strands-compatible tool objects.
# We pass those directly into Agent(tools=[...]).
#
# The system prompt tells the LLM what tools are available and
# when to use each one.
# -----------------------------------------------------------------

def build_agent(mcp_tools: list) -> Agent:
    """Create a Strands Agent loaded with the MCP server's tools."""
    system_prompt = (
        "You are a helpful AI assistant connected to a local MCP server.\n\n"
        "You have access to these MCP tools — use them when appropriate:\n"
        "  • calculator       – arithmetic and math (e.g. sqrt, powers)\n"
        "  • get_weather      – weather report for any city\n"
        "  • get_current_time – today's date and current time\n"
        "  • word_count       – count words/characters in any text\n\n"
        "Always call the right tool instead of guessing the answer.\n"
        "Be concise and friendly."
    )
    return Agent(
        model=ollama_model,
        system_prompt=system_prompt,
        tools=mcp_tools,             # tools discovered from the MCP server
    )


# -----------------------------------------------------------------
# SECTION 5 – INTERACTIVE CHAT LOOP
#
# The entire loop runs inside `with mcp_client:` so the MCP server
# subprocess is alive for the whole session and shut down cleanly
# when the user types 'quit'.
# -----------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Challenge 5 – MCP Chatbot")
    print("  Model   : llama3.2:3b  (Ollama, local)")
    print("  MCP     : local stdio server  (mcp_server.py)")
    print("  Commands: 'tools' = list MCP tools  |  'quit' = exit")
    print("=" * 60)

    # Open the MCP connection — this starts mcp_server.py as a
    # subprocess and keeps it running until the with block exits.
    with mcp_client:

        # Discover tools exposed by the MCP server
        print("\nConnecting to MCP server …")
        mcp_tools = mcp_client.list_tools_sync()
        print(f"Connected!  {len(mcp_tools)} tool(s) loaded from MCP server.")

        # Build the Strands Agent with those tools
        agent = build_agent(mcp_tools)

        print()
        print("Try asking:")
        print("  • What is sqrt(256) + 100?")
        print("  • What's the weather in Chennai?")
        print("  • What is today's date and time?")
        print("  • Count the words in: 'The quick brown fox jumps'")
        print()

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # ── exit ──────────────────────────────────────────
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # ── list MCP tools ─────────────────────────────────
            if user_input.lower() == "tools":
                print("\nMCP tools available to the agent:")
                for t in mcp_tools:
                    # Each tool object has a .tool_name attribute
                    name = getattr(t, "tool_name", str(t))
                    print(f"  • {name}")
                print()
                continue

            # ── normal turn ────────────────────────────────────
            # The agent decides which MCP tool to call (if any)
            # based on the user's message and the system prompt.
            print("\nAgent: ", end="", flush=True)
            agent(user_input)
            print()


# -----------------------------------------------------------------
# SECTION 6 – ENTRY POINT
# -----------------------------------------------------------------

if __name__ == "__main__":
    main()
