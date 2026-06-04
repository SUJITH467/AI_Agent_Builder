# =================================================================
# mcp_server.py  –  Local MCP Server for Challenge 5
#
# This file IS the MCP server. It runs as a separate process.
# starter.py launches it automatically via subprocess (stdio).
#
# It exposes four tools over the MCP protocol:
#   1. calculator        – evaluate a math expression
#   2. get_weather       – simulated weather for a city
#   3. get_current_time  – current date and time
#   4. word_count        – count words / characters in text
#
# You never run this file directly.
# starter.py starts it for you.
# =================================================================

from datetime import datetime
from mcp.server.fastmcp import FastMCP

# FastMCP is the simplest way to create an MCP server in Python.
# Give it a name — this appears in tool listings.
mcp = FastMCP("Challenge5-Local-MCP-Server")


# -----------------------------------------------------------------
# TOOL 1 – CALCULATOR
# The @mcp.tool() decorator registers this function as an MCP tool.
# The docstring becomes the tool description the LLM reads.
# -----------------------------------------------------------------
@mcp.tool()
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.
    Supports standard arithmetic and Python math functions like
    sqrt(), pow(), sin(), cos(), log(), etc.

    Args:
        expression: A Python math expression, e.g. '2 + 2' or 'sqrt(144)'.
    """
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# -----------------------------------------------------------------
# TOOL 2 – WEATHER
# -----------------------------------------------------------------
@mcp.tool()
def get_weather(city: str) -> str:
    """
    Return a weather report for a given city.
    Data is simulated — swap the dict for a real API call to go live.

    Args:
        city: City name, e.g. 'London', 'Tokyo', 'Chennai'.
    """
    mock = {
        "london":   "Cloudy, 15°C, Humidity 78%",
        "new york": "Sunny, 22°C, Humidity 55%",
        "tokyo":    "Humid, 28°C, Humidity 82%",
        "sydney":   "Windy, 19°C, Humidity 65%",
        "paris":    "Rainy, 17°C, Humidity 80%",
        "mumbai":   "Hot & Humid, 32°C, Humidity 88%",
        "chennai":  "Hot, 35°C, Humidity 75%",
        "dubai":    "Sunny & Hot, 38°C, Humidity 45%",
    }
    report = mock.get(city.lower().strip(), f"Partly Cloudy, 20°C, Humidity 60%")
    return f"Weather in {city.title()}: {report}"


# -----------------------------------------------------------------
# TOOL 3 – CURRENT TIME
# -----------------------------------------------------------------
@mcp.tool()
def get_current_time(timezone: str = "local") -> str:
    """
    Return the current date and time.

    Args:
        timezone: Pass 'local' (default) for the system clock.
                  This demo always returns local time.
    """
    now = datetime.now()
    return (
        f"Current date/time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}"
    )


# -----------------------------------------------------------------
# TOOL 4 – WORD COUNT
# -----------------------------------------------------------------
@mcp.tool()
def word_count(text: str) -> str:
    """
    Count the number of words and characters in the provided text.

    Args:
        text: Any string whose words and characters you want counted.
    """
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    return (
        f"Words: {words} | "
        f"Characters (with spaces): {chars} | "
        f"Characters (without spaces): {chars_no_spaces}"
    )


# -----------------------------------------------------------------
# Entry point – FastMCP runs the server over stdio when executed.
# starter.py launches this file as a subprocess and communicates
# with it through stdin / stdout using the MCP protocol.
# -----------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
