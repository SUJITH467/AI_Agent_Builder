# =============================================================
# Challenge 2 – Tools Agent
# Strands SDK + Ollama llama3.2:3b
# Tools: calculator, weather, age_calculator
# =============================================================

from datetime import date

from strands import Agent, tool
from strands.models.ollama import OllamaModel


# -------------------------------------------------------------
# SECTION 1 – TOOLS
# Every tool is a plain Python function decorated with @tool.
# The decorator reads the function name, docstring, and type
# hints to automatically build the schema the model needs so
# it knows when and how to call each tool.
# -------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this tool whenever the user asks for any calculation,
    arithmetic, or math problem.

    Args:
        expression: A valid Python math expression, e.g. '2 + 2',
                    '10 * (3 + 4)', 'sqrt(16)' (via math module).
    """
    import math  # available inside the expression via eval

    try:
        # eval is safe here because we control the environment:
        # only math functions and numeric literals are accessible.
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_weather(city: str) -> str:
    """
    Return a simulated weather report for a city.
    Use this tool when the user asks about the weather, temperature,
    or forecast for any location.

    Args:
        city: The name of the city, e.g. 'London', 'Tokyo'.
    """
    # In a real agent you would call a weather API here.
    # For this challenge we return realistic-looking mock data
    # so the agent can demonstrate tool-calling without needing
    # an API key.
    mock_data = {
        "london":   {"temp": "15°C", "condition": "Cloudy",  "humidity": "78%"},
        "new york": {"temp": "22°C", "condition": "Sunny",   "humidity": "55%"},
        "tokyo":    {"temp": "28°C", "condition": "Humid",   "humidity": "82%"},
        "sydney":   {"temp": "19°C", "condition": "Windy",   "humidity": "65%"},
        "paris":    {"temp": "17°C", "condition": "Rainy",   "humidity": "80%"},
    }

    key = city.lower().strip()
    if key in mock_data:
        w = mock_data[key]
        return (
            f"Weather in {city.title()}: {w['condition']}, "
            f"Temp: {w['temp']}, Humidity: {w['humidity']}"
        )
    # Default fallback so the tool always returns something useful
    return (
        f"Weather in {city.title()}: Partly Cloudy, Temp: 20°C, Humidity: 60%"
    )


@tool
def age_calculator(birth_year: int, birth_month: int, birth_day: int) -> str:
    """
    Calculate a person's exact age from their date of birth.
    Use this tool whenever the user asks how old someone is or
    wants to calculate an age from a birth date.

    Args:
        birth_year:  Four-digit birth year, e.g. 1990.
        birth_month: Birth month as an integer (1 = January, 12 = December).
        birth_day:   Birth day as an integer (1-31).
    """
    try:
        today = date.today()
        birthdate = date(birth_year, birth_month, birth_day)

        if birthdate > today:
            return "That birth date is in the future — please check the date."

        # Calculate full years elapsed
        age = today.year - birthdate.year

        # Subtract 1 if the birthday hasn't occurred yet this year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            age -= 1

        # Days until next birthday
        next_birthday = birthdate.replace(year=today.year)
        if next_birthday < today:
            next_birthday = next_birthday.replace(year=today.year + 1)
        days_left = (next_birthday - today).days

        return (
            f"Age: {age} years old. "
            f"Next birthday in {days_left} day(s) "
            f"({next_birthday.strftime('%B %d, %Y')})."
        )
    except ValueError as e:
        return f"Invalid date: {e}"


# -------------------------------------------------------------
# SECTION 2 – MODEL + AGENT SETUP
# We create the OllamaModel and pass all three tools to the
# Agent at initialization time.
# -------------------------------------------------------------

ollama_model = OllamaModel(
    host="http://localhost:11434",   # default Ollama address
    model_id="llama3.2:3b",
    temperature=0.3,  # lower = more precise for tool-heavy tasks
)

agent = Agent(
    model=ollama_model,
    # Giving the agent a clear system prompt helps it decide
    # when to use tools vs. answering from its own knowledge.
    system_prompt=(
        "You are a helpful assistant with access to three tools:\n"
        "1. calculator   – use it for any maths or arithmetic\n"
        "2. get_weather  – use it for weather questions\n"
        "3. age_calculator – use it to calculate someone's age\n"
        "Always use the appropriate tool when the question calls for it."
    ),
    tools=[calculator, get_weather, age_calculator],
)


# -------------------------------------------------------------
# SECTION 3 – INTERACTIVE REPL
# A simple loop so you can ask multiple questions in one session.
# The agent keeps conversation history automatically, so follow-up
# questions ("what about tomorrow?") work without re-typing context.
# -------------------------------------------------------------

def main():
    print("=" * 55)
    print("  Challenge 2 – Tools Agent")
    print("  Model  : llama3.2:3b (Ollama, local)")
    print("  Tools  : calculator | get_weather | age_calculator")
    print("  Type 'quit' to exit  |  'tools' to list tools")
    print("=" * 55)
    print()
    print("Try asking:")
    print("  • What is 25 * 48 + 100?")
    print("  • What's the weather in Tokyo?")
    print("  • How old is someone born on March 15, 1990?")
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Handy shortcut so beginners can see what tools are loaded
        if user_input.lower() == "tools":
            print("Loaded tools:", agent.tool_names)
            print()
            continue

        print("\nAgent: ", end="", flush=True)
        agent(user_input)
        print()


if __name__ == "__main__":
    main()
