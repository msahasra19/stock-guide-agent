import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import yfinance as yf
import requests

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Add NEWS_API_KEY to your .env

# --- Tools ---

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's initial stock query to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}


def get_stock_data(ticker: str) -> dict:
    """
    Fetches fundamental stock data for a given ticker symbol using yfinance.
    Returns price, market cap, P/E ratio, 52-week range, dividend yield, and analyst recommendation.

    Args:
        ticker: The stock ticker symbol (e.g. 'AAPL', 'TSLA', 'MSFT').

    Returns:
        A dict with stock fundamentals and analyst recommendation, or an error message.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return {"error": f"No data found for ticker '{ticker}'. Please verify the symbol."}

        recommendation = info.get("recommendationKey", "n/a").upper()

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", "N/A"),
            "current_price": info.get("regularMarketPrice"),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "analyst_recommendation": recommendation,  # STRONG_BUY / BUY / HOLD / SELL etc.
            "summary": info.get("longBusinessSummary", "N/A"),
        }
    except Exception as e:
        logging.error(f"yfinance error for {ticker}: {e}")
        return {"error": str(e)}


def get_stock_news(query: str) -> dict:
    """
    Fetches the latest news articles related to a stock or market topic using NewsAPI.

    Args:
        query: A search term such as a company name, ticker, or market topic (e.g. 'Apple stock', 'Fed interest rates').

    Returns:
        A dict containing a list of recent news articles with title, source, and URL.
    """
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={requests.utils.quote(query)}"
            f"&sortBy=publishedAt"
            f"&pageSize=5"
            f"&language=en"
            f"&apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = [
            {
                "title": a["title"],
                "source": a["source"]["name"],
                "published_at": a["publishedAt"],
                "url": a["url"],
                "description": a.get("description", ""),
            }
            for a in data.get("articles", [])
        ]
        return {"query": query, "articles": articles}
    except Exception as e:
        logging.error(f"NewsAPI error for '{query}': {e}")
        return {"error": str(e)}


# --- Langchain Tools ---

wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# --- Agents ---

# 1. Researcher Agent
stock_researcher = Agent(
    name="stock_researcher",
    model=model_name,
    description=(
        "Gathers comprehensive stock data from yfinance (fundamentals & analyst signals), "
        "NewsAPI (live news), and Wikipedia (concepts & background)."
    ),
    instruction="""
    You are a senior financial research assistant. Your goal is to fully investigate the user's PROMPT.

    You have access to three tools:
    1. get_stock_data  — fetches real-time fundamentals (price, P/E, market cap, 52-week range,
       dividend yield, sector, and analyst recommendation) for a given ticker.
    2. get_stock_news  — fetches the latest news articles for a company or market topic.
    3. wikipedia_tool  — looks up background knowledge (e.g. what a P/E ratio means, company history,
       macroeconomic concepts).

    Instructions:
    - Identify any ticker symbols or company names in the PROMPT and call get_stock_data for each.
    - Call get_stock_news with a relevant query derived from the PROMPT.
    - If the PROMPT asks about a concept (e.g. "what is a P/E ratio?"), use wikipedia_tool.
    - You MUST use all relevant tools. Do not skip a tool if it would help answer the question.
    - Output a structured summary of all findings: fundamentals, news highlights, and concept explanations.
    - Do NOT yet give a final recommendation — just compile the raw research.

    PROMPT:
    { PROMPT }
    """,
    tools=[get_stock_data, get_stock_news, wikipedia_tool],
    output_key="research_data",
)

# 2. Analyst / Formatter Agent
stock_analyst = Agent(
    name="stock_analyst",
    model=model_name,
    description="Synthesises research into a clear, structured stock guide response with a recommendation.",
    instruction="""
    You are a friendly but professional stock market guide. Your task is to take RESEARCH_DATA
    and present it as a clear, well-structured response to the user.

    Structure your response as follows:

    1. **Overview** — Company name, sector, current price, and a one-line description.
    2. **Key Fundamentals** — Market cap, P/E ratio, 52-week range, dividend yield.
    3. **Latest News** — Summarise 2–3 of the most relevant recent headlines and their implications.
    4. **Concept Explanation** (if applicable) — If the user asked about a market concept, explain it clearly.
    5. **Analyst Verdict** — Based on the aggregated analyst recommendation and the news sentiment,
       state whether the stock leans toward BUY / HOLD / SELL and briefly explain why.

    Important:
    - Always include a disclaimer: "This is not financial advice. Please consult a qualified financial
      advisor before making investment decisions."
    - Be conversational, clear, and avoid unnecessary jargon.
    - If data is missing for any field, skip it gracefully.

    RESEARCH_DATA:
    { research_data }
    """,
)

# --- Workflow ---

stock_guide_workflow = SequentialAgent(
    name="stock_guide_workflow",
    description="Researches a stock query then formats a comprehensive, recommendation-ready response.",
    sub_agents=[
        stock_researcher,  # Step 1: Gather fundamentals, news, and concepts
        stock_analyst,     # Step 2: Synthesise and present with a verdict
    ],
)

# --- Root Agent ---

root_agent = Agent(
    name="stock_guide_greeter",
    model=model_name,
    description="Entry point for the Stock Guide. Greets the user and captures their query.",
    instruction="""
    You are a friendly stock market guide assistant.
    - Greet the user and let them know you can help them with:
        * Stock fundamentals and analysis (just give you a ticker or company name)
        * Latest market news
        * Buy / Hold / Sell signals based on analyst consensus
        * Explanations of stock market concepts
    - When the user responds with their query, use the 'add_prompt_to_state' tool to save it.
    - After saving, transfer control to the 'stock_guide_workflow' agent immediately.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[stock_guide_workflow],
)