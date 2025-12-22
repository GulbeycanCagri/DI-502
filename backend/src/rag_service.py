import asyncio
import datetime
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# --- LlamaIndex & Ollama Imports ---
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Memory Manager Import
from backend.src.memory_manager import memory_manager

load_dotenv()

# --- API Keys ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Optional: newsapi.org
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # Optional: Alpha Vantage


class QueryType(Enum):
    """Types of financial queries we can handle."""

    PRICE_LOOKUP = "price_lookup"  # Direct price query: "What is X price?"
    PRICE_ANALYSIS = "price_analysis"  # Analysis/prediction: "Will X price increase?"
    NEWS_QUERY = "news_query"  # News request: "What's happening with X?"
    MARKET_ANALYSIS = "market_analysis"  # Market trends: "How is the market?"
    COMPANY_INFO = "company_info"  # Company details: "Tell me about X"
    GENERAL_FINANCE = "general_finance"  # General questions
    UNKNOWN = "unknown"


@dataclass
class QueryIntent:
    """Represents the parsed intent of a user query."""

    query_type: QueryType
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    crypto_symbol: Optional[str] = None
    keywords: List[str] = None
    needs_price_data: bool = False  # Should we fetch real-time price?
    needs_news: bool = True  # Should we fetch news? (default: always yes)
    is_prediction_question: bool = False  # Is this asking for prediction/opinion?

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


# Ollama
llm = Ollama(model="llama3", request_timeout=300.0, temperature=0.1)

# Embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024


# --- Extended Ticker Mapping ---
# More comprehensive mapping with common variations
TICKER_MAPPING: Dict[str, str] = {
    # Tech Giants
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "facebook": "META",
    "fb": "META",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "netflix": "NFLX",
    "nflx": "NFLX",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "intel": "INTC",
    "intc": "INTC",
    # Financial
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "jpm": "JPM",
    "goldman sachs": "GS",
    "goldman": "GS",
    "gs": "GS",
    "bank of america": "BAC",
    "bofa": "BAC",
    "bac": "BAC",
    "visa": "V",
    "mastercard": "MA",
    "paypal": "PYPL",
    "pypl": "PYPL",
    "berkshire": "BRK.B",
    "berkshire hathaway": "BRK.B",
    # Retail & Consumer
    "walmart": "WMT",
    "wmt": "WMT",
    "costco": "COST",
    "cost": "COST",
    "disney": "DIS",
    "walt disney": "DIS",
    "dis": "DIS",
    "nike": "NKE",
    "nke": "NKE",
    "starbucks": "SBUX",
    "sbux": "SBUX",
    "coca cola": "KO",
    "coke": "KO",
    "ko": "KO",
    "mcdonalds": "MCD",
    "mcd": "MCD",
    # Healthcare & Pharma
    "johnson & johnson": "JNJ",
    "j&j": "JNJ",
    "jnj": "JNJ",
    "pfizer": "PFE",
    "pfe": "PFE",
    "moderna": "MRNA",
    "mrna": "MRNA",
    "unitedhealth": "UNH",
    "unh": "UNH",
    # Energy
    "exxon": "XOM",
    "exxon mobil": "XOM",
    "xom": "XOM",
    "chevron": "CVX",
    "cvx": "CVX",
    # Others
    "salesforce": "CRM",
    "crm": "CRM",
    "adobe": "ADBE",
    "adbe": "ADBE",
    "oracle": "ORCL",
    "orcl": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
    "csco": "CSCO",
    "boeing": "BA",
    "ba": "BA",
    "uber": "UBER",
    "airbnb": "ABNB",
    "abnb": "ABNB",
    "spotify": "SPOT",
    "spot": "SPOT",
    "zoom": "ZM",
    "zm": "ZM",
    "snowflake": "SNOW",
    "snow": "SNOW",
    "palantir": "PLTR",
    "pltr": "PLTR",
}

# Cryptocurrency mapping
CRYPTO_MAPPING: Dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "cardano": "ADA",
    "ada": "ADA",
    "ripple": "XRP",
    "xrp": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "polkadot": "DOT",
    "dot": "DOT",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "chainlink": "LINK",
    "link": "LINK",
    "polygon": "MATIC",
    "matic": "MATIC",
    "litecoin": "LTC",
    "ltc": "LTC",
    "shiba inu": "SHIB",
    "shib": "SHIB",
    "uniswap": "UNI",
    "uni": "UNI",
    "binance coin": "BNB",
    "bnb": "BNB",
}


def _build_chat_messages(
    session_id: Optional[str], question: str, system_prompt: str
) -> List[ChatMessage]:
    """
    Build chat messages list with conversation history.

    Args:
        session_id: Session ID for memory retrieval
        question: Current user question
        system_prompt: System instructions

    Returns:
        List of ChatMessage objects for the LLM
    """
    messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]

    # Add conversation history if session exists
    if session_id:
        history = memory_manager.get_chat_history(session_id)
        messages.extend(history)

    # Add current user message
    messages.append(ChatMessage(role=MessageRole.USER, content=question))

    return messages


# --- Query Intent Analysis ---


def analyze_query_intent_with_llm(question: str) -> Dict[str, Any]:
    """
    Use LLM to analyze query intent - this is the PRIMARY method for intent detection.
    Provides nuanced understanding of user questions.

    Args:
        question: User's question

    Returns:
        Dictionary with comprehensive intent analysis
    """
    prompt = (
        """You are a financial query analyzer. Analyze the question and respond ONLY with valid JSON.

Determine:
1. "intent": The type of query
   - "price_lookup": Direct price question (e.g., "What is Bitcoin price?", "NVDA stock price")
   - "price_analysis": Analysis/prediction about price (e.g., "Will Bitcoin increase?", "Is AAPL overvalued?")
   - "news_query": News/updates request (e.g., "Latest Tesla news", "What's happening with crypto?")
   - "market_analysis": Market trends/overview (e.g., "How is the market?", "Tech sector outlook")
   - "company_info": Company information (e.g., "Tell me about Apple", "What does NVIDIA do?")
   - "general_finance": Other finance questions

2. "ticker": Stock symbol if mentioned (e.g., "NVDA", "AAPL") or null
3. "crypto": Crypto symbol if mentioned (e.g., "BTC", "ETH") or null
4. "company": Company name if mentioned or null
5. "needs_price": true if answering requires current price data
6. "needs_news": true if answering requires recent news (usually true for analysis)
7. "is_prediction": true if asking for future prediction/opinion

Examples:
Q: "What is NVIDIA stock price?"
A: {"intent":"price_lookup","ticker":"NVDA","crypto":null,"company":"NVIDIA","needs_price":true,"needs_news":false,"is_prediction":false}

Q: "Do you think Bitcoin price will increase?"
A: {"intent":"price_analysis","ticker":null,"crypto":"BTC","company":null,"needs_price":true,"needs_news":true,"is_prediction":true}

Q: "What's happening with Tesla lately?"
A: {"intent":"news_query","ticker":"TSLA","crypto":null,"company":"Tesla","needs_price":false,"needs_news":true,"is_prediction":false}

Q: "Is Apple stock overvalued?"
A: {"intent":"price_analysis","ticker":"AAPL","crypto":null,"company":"Apple","needs_price":true,"needs_news":true,"is_prediction":true}

Q: "How is the crypto market doing?"
A: {"intent":"market_analysis","ticker":null,"crypto":null,"company":null,"needs_price":false,"needs_news":true,"is_prediction":false}

Question: """
        + question
        + """
JSON:"""
    )

    try:
        response = llm.complete(prompt)
        response_text = str(response).strip()
        print("LLM", response_text)

        import json

        # Find JSON in response
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            print(f"LLM Intent Analysis: {result}")
            return result
    except Exception as e:
        print(f"LLM intent analysis failed: {e}")

    return {}


def extract_entities_rule_based(
    question: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fast rule-based extraction for ticker, crypto, and company names.
    Used as a fallback/supplement to LLM analysis.

    Args:
        question: User's question (lowercase)

    Returns:
        Tuple of (ticker, crypto_symbol, company_name)
    """
    lowered = question.lower()
    ticker = None
    crypto = None
    company = None

    for company_name, tick in sorted(TICKER_MAPPING.items(), key=lambda x: -len(x[0])):
        if company_name in lowered:
            ticker = tick
            if len(company_name) > 3:
                company = company_name.title()
            break

    for crypto_name, symbol in sorted(CRYPTO_MAPPING.items(), key=lambda x: -len(x[0])):
        if crypto_name in lowered:
            crypto = symbol
            break

    if not ticker:
        ticker_pattern = r"\$?([A-Z]{1,5})(?:\s+stock|\s+share)?"
        matches = re.findall(ticker_pattern, question.upper())
        for match in matches:
            if match in TICKER_MAPPING.values():
                ticker = match
                break

    return ticker, crypto, company


def analyze_query_intent(question: str) -> QueryIntent:
    """
    Analyze the user's question using LLM-first approach.
    Rule-based extraction is only used as fallback for entity extraction.

    Args:
        question: User's question

    Returns:
        QueryIntent object with comprehensive parsed information
    """
    print(f"--- Analyzing query intent for: '{question}' ---")

    llm_result = analyze_query_intent_with_llm(question)

    rule_ticker, rule_crypto, rule_company = extract_entities_rule_based(question)

    intent = QueryIntent(query_type=QueryType.GENERAL_FINANCE)

    # Use LLM results if available, otherwise fall back to rules
    if llm_result:
        # Map LLM intent to QueryType
        intent_mapping = {
            "price_lookup": QueryType.PRICE_LOOKUP,
            "price_analysis": QueryType.PRICE_ANALYSIS,
            "news_query": QueryType.NEWS_QUERY,
            "market_analysis": QueryType.MARKET_ANALYSIS,
            "company_info": QueryType.COMPANY_INFO,
            "general_finance": QueryType.GENERAL_FINANCE,
        }
        llm_intent = llm_result.get("intent", "general_finance")
        intent.query_type = intent_mapping.get(llm_intent, QueryType.GENERAL_FINANCE)

        # Get entities - prefer LLM but fall back to rule-based
        # Handle case where LLM returns a list instead of string
        llm_ticker = llm_result.get("ticker")
        llm_crypto = llm_result.get("crypto")
        llm_company = llm_result.get("company")

        # Convert lists to first element (take primary asset)
        if isinstance(llm_ticker, list):
            llm_ticker = llm_ticker[0] if llm_ticker else None
        if isinstance(llm_crypto, list):
            llm_crypto = llm_crypto[0] if llm_crypto else None
        if isinstance(llm_company, list):
            llm_company = llm_company[0] if llm_company else None

        intent.ticker = llm_ticker or rule_ticker
        intent.crypto_symbol = llm_crypto or rule_crypto
        intent.company_name = llm_company or rule_company

        # Get flags from LLM
        intent.needs_price_data = llm_result.get("needs_price", False)
        intent.needs_news = llm_result.get("needs_news", True)  # Default to true
        intent.is_prediction_question = llm_result.get("is_prediction", False)
    else:
        # Full fallback to rule-based
        print("--- Falling back to rule-based intent detection ---")
        intent.ticker = rule_ticker
        intent.crypto_symbol = rule_crypto
        intent.company_name = rule_company

        # Simple heuristics for fallback
        lowered = question.lower()
        if any(w in lowered for w in ["price", "cost", "worth", "trading at"]):
            if any(
                w in lowered
                for w in [
                    "will",
                    "think",
                    "predict",
                    "increase",
                    "decrease",
                    "go up",
                    "go down",
                ]
            ):
                intent.query_type = QueryType.PRICE_ANALYSIS
                intent.needs_price_data = True
                intent.needs_news = True
                intent.is_prediction_question = True
            else:
                intent.query_type = QueryType.PRICE_LOOKUP
                intent.needs_price_data = True
                intent.needs_news = False
        elif any(w in lowered for w in ["news", "happening", "latest", "recent"]):
            intent.query_type = QueryType.NEWS_QUERY
            intent.needs_news = True
        else:
            intent.query_type = QueryType.GENERAL_FINANCE
            intent.needs_news = True

    intent.keywords = generate_search_keywords(question)

    print(
        f"Final Intent: type={intent.query_type.value}, ticker={intent.ticker}, "
        f"crypto={intent.crypto_symbol}, needs_price={intent.needs_price_data}, "
        f"needs_news={intent.needs_news}, is_prediction={intent.is_prediction_question}"
    )

    return intent


def extract_ticker_from_question(question: str) -> Optional[str]:
    """
    Extract stock ticker from question using extended mapping.
    DEPRECATED: Use extract_entities_rule_based instead.
    Kept for backward compatibility.
    """
    ticker, _, _ = extract_entities_rule_based(question)
    return ticker


def extract_crypto_from_question(question: str) -> Optional[str]:
    """
    Extract cryptocurrency symbol from question.
    DEPRECATED: Use extract_entities_rule_based instead.
    Kept for backward compatibility.
    """
    _, crypto, _ = extract_entities_rule_based(question)
    return crypto


def generate_search_keywords(question: str) -> List[str]:
    """
    Generate optimized search keywords from a question using LLM.

    Args:
        question: User's question

    Returns:
        List of search keywords
    """
    print("--- Generating search keywords ---")
    prompt = (
        "You are a search query generator for financial news. "
        "Extract 3-5 important keywords from the following question. "
        "Focus on company names, financial terms, and specific topics. "
        "IMPORTANT: Output ONLY the keywords separated by commas. No other text.\n\n"
        f"Question: {question}\nKeywords:"
    )
    try:
        response = llm.complete(prompt)
        keywords_str = str(response).strip().split("\n")[0].replace('"', "")
        keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        print(f"Generated keywords: {keywords}")
        return keywords[:5]  # Limit to 5 keywords
    except Exception as e:
        print(f"Keyword generation failed: {e}. Using fallback.")
        # Fallback: simple word extraction
        words = question.lower().split()
        stop_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "of",
            "for",
            "in",
            "on",
            "to",
            "with",
            "how",
            "why",
            "when",
            "where",
            "can",
            "you",
            "me",
            "about",
        }
        return [w for w in words if w not in stop_words and len(w) > 2][:5]


async def plain_chat(
    question: str, test: bool = False, session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Plain chat with Ollama (Async Streaming) with conversation memory.

    Args:
        question: User's question
        test: Whether to run in test mode
        session_id: Optional session ID for conversation memory
    """
    if test:
        yield "Test response from Ollama"
        return

    print(f"--- Performing plain chat with Ollama (Async) | Session: {session_id} ---")

    system_prompt = (
        "You are a financial analyst providing investor-focused insights. Based on the conversation context, "
        "identify and explain information material to investment decisions. If relevant information is absent, indicate this. "
        "Do not use context reference labels in your answer. Never use hashtags or emojis. "
        "Remember previous messages in the conversation to provide contextually relevant answers."
    )

    try:
        # Build messages with history
        messages = _build_chat_messages(session_id, question, system_prompt)

        # Store user message in memory
        if session_id:
            memory_manager.add_message(session_id, "user", question)

        # Stream the response
        full_response = ""
        response_gen = await llm.astream_chat(messages)

        async for chunk in response_gen:
            delta = chunk.delta if hasattr(chunk, "delta") else str(chunk)
            full_response += delta
            yield delta
            await asyncio.sleep(0)

        # Store assistant response in memory
        if session_id and full_response:
            memory_manager.add_message(session_id, "assistant", full_response)

    except Exception as e:
        print(f"Ollama Chat Error: {e}")
        yield f"Error: {str(e)}"


def fetch_stock_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current stock quote from Finnhub.

    Args:
        ticker: Stock ticker symbol - must be a string

    Returns:
        Dictionary with stock price data or None
    """
    # Handle case where ticker might be a list or None
    if ticker is None:
        return None
    if isinstance(ticker, list):
        ticker = ticker[0] if ticker else None
        if not ticker:
            return None
    if not isinstance(ticker, str):
        print(f"Invalid ticker type: {type(ticker)}, value: {ticker}")
        return None

    try:
        endpoint = "https://finnhub.io/api/v1/quote"
        params = {"symbol": ticker, "token": FINNHUB_API_KEY}
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("c", 0) > 0:  # 'c' is current price
            return {
                "current_price": data.get("c"),
                "change": data.get("d"),
                "percent_change": data.get("dp"),
                "high": data.get("h"),
                "low": data.get("l"),
                "open": data.get("o"),
                "previous_close": data.get("pc"),
            }
    except Exception as e:
        print(f"Error fetching stock quote for {ticker}: {e}")
    return None


def fetch_company_profile(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch company profile from Finnhub.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with company info or None
    """
    try:
        endpoint = "https://finnhub.io/api/v1/stock/profile2"
        params = {"symbol": ticker, "token": FINNHUB_API_KEY}
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("name"):
            return {
                "name": data.get("name"),
                "ticker": data.get("ticker"),
                "industry": data.get("finnhubIndustry"),
                "market_cap": data.get("marketCapitalization"),
                "website": data.get("weburl"),
                "country": data.get("country"),
            }
    except Exception as e:
        print(f"Error fetching company profile for {ticker}: {e}")
    return None


def fetch_company_news(ticker: str, days: int = 15) -> List[Dict[str, Any]]:
    """
    Fetch company-specific news from Finnhub.

    Args:
        ticker: Stock ticker symbol
        days: Number of days to look back

    Returns:
        List of news articles
    """
    try:
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=days)
        endpoint = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "token": FINNHUB_API_KEY,
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()[:25]
    except Exception as e:
        print(f"Error fetching company news for {ticker}: {e}")
    return []


def fetch_general_news(
    category: str = "general", days: int = 15
) -> List[Dict[str, Any]]:
    """
    Fetch general financial news from Finnhub.

    Args:
        category: News category (general, forex, crypto, merger)

    Returns:
        List of news articles
    """
    try:
        endpoint = "https://finnhub.io/api/v1/news"
        from_date = datetime.date.today() - datetime.timedelta(days=days)
        params = {
            "category": category,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": datetime.date.today().strftime("%Y-%m-%d"),
            "minId": 0,
            "token": FINNHUB_API_KEY,
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()[:30]
    except Exception as e:
        print(f"Error fetching general news: {e}")
    return []


def fetch_crypto_price(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch cryptocurrency price from CoinGecko (free API, no key needed).

    Args:
        symbol: Crypto symbol (e.g., BTC, ETH) - must be a string

    Returns:
        Dictionary with crypto price data or None
    """
    # Handle case where symbol might be a list or None
    if symbol is None:
        return None
    if isinstance(symbol, list):
        symbol = symbol[0] if symbol else None
        if not symbol:
            return None
    if not isinstance(symbol, str):
        print(f"Invalid symbol type: {type(symbol)}, value: {symbol}")
        return None

    # CoinGecko uses different IDs
    coingecko_ids = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "XRP": "ripple",
        "DOGE": "dogecoin",
        "DOT": "polkadot",
        "AVAX": "avalanche-2",
        "LINK": "chainlink",
        "MATIC": "matic-network",
        "LTC": "litecoin",
        "SHIB": "shiba-inu",
        "UNI": "uniswap",
        "BNB": "binancecoin",
    }

    coin_id = coingecko_ids.get(symbol.upper())
    if not coin_id:
        return None

    try:
        endpoint = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if coin_id in data:
            coin_data = data[coin_id]
            return {
                "symbol": symbol,
                "price_usd": coin_data.get("usd"),
                "change_24h": coin_data.get("usd_24h_change"),
                "market_cap": coin_data.get("usd_market_cap"),
            }
    except Exception as e:
        print(f"Error fetching crypto price for {symbol}: {e}")
    return None


def fetch_crypto_news() -> List[Dict[str, Any]]:
    """
    Fetch cryptocurrency news from Finnhub.

    Returns:
        List of crypto news articles
    """
    return fetch_general_news(category="crypto")


def fetch_market_news_newsapi(keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch news from NewsAPI.org (optional, requires API key).
    Provides broader news coverage beyond Finnhub.

    Args:
        keywords: Search keywords

    Returns:
        List of news articles
    """
    if not NEWS_API_KEY:
        return []

    try:
        query = " OR ".join(keywords[:3])  # NewsAPI has query limits
        endpoint = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": NEWS_API_KEY,
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return [
            {
                "headline": article.get("title"),
                "summary": article.get("description"),
                "source": article.get("source", {}).get("name"),
                "url": article.get("url"),
                "datetime": article.get("publishedAt"),
            }
            for article in data.get("articles", [])
        ]
    except Exception as e:
        print(f"Error fetching NewsAPI news: {e}")
    return []


def build_context_from_data(
    intent: QueryIntent,
    stock_quote: Optional[Dict] = None,
    company_profile: Optional[Dict] = None,
    crypto_price: Optional[Dict] = None,
    news_articles: List[Dict] = None,
) -> str:
    """
    Build a context string from fetched data for the LLM.

    Args:
        intent: Query intent
        stock_quote: Stock price data
        company_profile: Company info
        crypto_price: Crypto price data
        news_articles: List of news articles

    Returns:
        Formatted context string
    """
    context_parts = []

    if stock_quote and intent.ticker:
        # Handle None values safely
        current_price = stock_quote.get("current_price", 0) or 0
        change = stock_quote.get("change", 0) or 0
        percent_change = stock_quote.get("percent_change", 0) or 0
        high = stock_quote.get("high", 0) or 0
        low = stock_quote.get("low", 0) or 0
        open_price = stock_quote.get("open", 0) or 0
        prev_close = stock_quote.get("previous_close", 0) or 0

        company_name = intent.company_name or intent.ticker
        context_parts.append(f"""
=== REAL-TIME STOCK DATA FOR {company_name} ({intent.ticker}) ===
** IMPORTANT: Use this exact data to answer price questions **
Current Price: ${current_price:.2f} USD
Price Change: ${change:+.2f} ({percent_change:+.2f}%)
Day High: ${high:.2f}
Day Low: ${low:.2f}
Open: ${open_price:.2f}
Previous Close: ${prev_close:.2f}
===================================================
""")

    if company_profile:
        market_cap = company_profile.get("market_cap", 0) or 0
        context_parts.append(f"""
=== Company Profile ===
Name: {company_profile.get('name', 'N/A')}
Industry: {company_profile.get('industry', 'N/A')}
Market Cap: ${market_cap:,.0f}M
Country: {company_profile.get('country', 'N/A')}
""")

    if crypto_price:
        price_usd = crypto_price.get("price_usd", 0) or 0
        change_24h = crypto_price.get("change_24h", 0) or 0
        market_cap = crypto_price.get("market_cap", 0) or 0

        context_parts.append(f"""
=== REAL-TIME CRYPTOCURRENCY DATA FOR {crypto_price.get('symbol', 'N/A')} ===
** IMPORTANT: Use this exact data to answer price questions **
Current Price: ${price_usd:,.2f} USD
24h Change: {change_24h:+.2f}%
Market Cap: ${market_cap:,.0f}
===================================================
""")

    if news_articles:
        context_parts.append("\n=== Recent News (for context, NOT for price data) ===")
        for i, article in enumerate(news_articles[:10], 1):
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            source = article.get("source", "Unknown")
            context_parts.append(f"\n{i}. [{source}] {headline}\n   {summary}")

    return "\n".join(context_parts)


async def query_online(
    question: str, test: bool = False, session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Enhanced online research with intelligent intent-based data fetching.
    Always fetches relevant news for context. Supports stocks, crypto, and general finance.

    Args:
        question: User's question
        test: Whether to run in test mode
        session_id: Optional session ID for conversation memory
    """
    if test:
        yield "Test online research response"
        return

    print(f"--- Performing enhanced online research | Session: {session_id} ---")

    # Analyze query intent using LLM-first approach
    intent = analyze_query_intent(question)

    try:
        documents_list = []
        additional_context = ""
        stock_quote = None
        company_profile = None
        crypto_price = None
        news_articles = []

        if intent.needs_price_data:
            if intent.ticker:
                print(f"Fetching stock quote for {intent.ticker}")
                stock_quote = fetch_stock_quote(intent.ticker)
                company_profile = fetch_company_profile(intent.ticker)
            elif intent.crypto_symbol:
                print(f"Fetching crypto price for {intent.crypto_symbol}")
                crypto_price = fetch_crypto_price(intent.crypto_symbol)

        if intent.needs_news:
            if intent.ticker:
                # Company-specific news
                print(f"Fetching news for {intent.ticker}")
                news_articles = fetch_company_news(intent.ticker, days=15)
            elif intent.crypto_symbol:
                # Crypto news
                print("Fetching crypto news")
                news_articles = fetch_crypto_news()
            else:
                # General market news
                print("Fetching general market news")
                general_news = fetch_general_news("general", days=7)
                merger_news = fetch_general_news("merger", days=7)
                news_articles = general_news + merger_news

            # Also fetch from NewsAPI if available and we have keywords
            if intent.keywords:
                newsapi_articles = fetch_market_news_newsapi(intent.keywords)
                # Convert to same format
                for article in newsapi_articles:
                    news_articles.append(
                        {
                            "headline": article.get("headline"),
                            "summary": article.get("summary"),
                            "source": article.get("source"),
                            "url": article.get("url"),
                        }
                    )

        additional_context = build_context_from_data(
            intent,
            stock_quote=stock_quote,
            company_profile=company_profile,
            crypto_price=crypto_price,
            news_articles=news_articles
            if not intent.needs_news
            else None,  # News goes to documents
        )

        if news_articles:
            for article in news_articles:
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                source_name = article.get("source", "Unknown")
                url = article.get("url", "")

                if headline:  # At minimum need headline
                    text_content = f"Source: {source_name}\nHeadline: {headline}"
                    if summary:
                        text_content += f"\nSummary: {summary}"

                    doc = Document(
                        text=text_content,
                        metadata={
                            "source_title": headline,
                            "url": url,
                            "source_name": source_name,
                        },
                    )
                    documents_list.append(doc)

        if not documents_list and not additional_context:
            yield "No relevant financial data or news found for your query. Please try rephrasing or ask about a different topic."
            return

        # Store user message in memory
        if session_id:
            memory_manager.add_message(session_id, "user", question)

        # Build conversation context
        conversation_context = ""
        if session_id:
            history_str = memory_manager.get_history_as_string(session_id)
            if history_str:
                conversation_context = f"\nPrevious conversation:\n{history_str}\n"

        if intent.query_type == QueryType.PRICE_LOOKUP:
            instruction = """
** INSTRUCTION: PRICE LOOKUP **
The user wants to know the current price. Use the REAL-TIME DATA provided above.
Start your answer with the exact current price.
"""
        elif intent.query_type == QueryType.PRICE_ANALYSIS:
            instruction = """
** INSTRUCTION: PRICE ANALYSIS/PREDICTION **
The user is asking for analysis or prediction about price movement.
1. First, mention the current price from the data above (if available).
2. Then, analyze the recent news to provide context.
3. Give your analysis based on news sentiment and market conditions.
4. Clearly state that this is analysis, not financial advice.
"""
        elif intent.query_type == QueryType.NEWS_QUERY:
            instruction = """
** INSTRUCTION: NEWS SUMMARY **
The user wants to know recent news and developments.
Summarize the most relevant and recent news from the articles provided.
Focus on significant events, announcements, and market-moving information.
"""
        elif intent.query_type == QueryType.MARKET_ANALYSIS:
            instruction = """
** INSTRUCTION: MARKET ANALYSIS **
Provide an overview of current market conditions based on the news provided.
Identify trends, significant events, and overall market sentiment.
"""
        else:
            instruction = """
** INSTRUCTION **
Answer the user's financial question based on the data and news provided.
Be specific and cite relevant information from the context.
"""

        enhanced_prompt = f"""You are a knowledgeable financial analyst providing accurate, data-driven insights.

{additional_context}
{instruction}
{conversation_context}

Based on the real-time data and recent news context provided, answer the following question.

IMPORTANT GUIDELINES:
- If price data is provided above, USE IT - never say you don't have price information.
- For analysis questions, use news to support your points.
- Be specific with numbers when available.
- For predictions/opinions, base them on the news sentiment and clearly state they are analysis, not advice.

Question: {question}"""

        print("Building index and synthesizing answer...")

        full_response = ""

        if documents_list:
            index = VectorStoreIndex.from_documents(documents_list)
            query_engine = index.as_query_engine(
                response_mode="compact",
                similarity_top_k=7,  # Increased for better context
                streaming=True,
            )

            streaming_response = await query_engine.aquery(enhanced_prompt)

            async for text in streaming_response.async_response_gen():
                full_response += text
                yield text
                await asyncio.sleep(0)
        else:
            # No news documents, use direct LLM response with price data
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a financial analyst. Provide accurate information based on the data given. Never say you don't have information if data is provided in the context.",
                ),
                ChatMessage(role=MessageRole.USER, content=enhanced_prompt),
            ]

            response_gen = await llm.astream_chat(messages)
            async for chunk in response_gen:
                delta = chunk.delta if hasattr(chunk, "delta") else str(chunk)
                full_response += delta
                yield delta
                await asyncio.sleep(0)

        # Store assistant response
        if session_id and full_response:
            memory_manager.add_message(session_id, "assistant", full_response)

    except Exception as e:
        print(f"Error in query_online: {e}")
        import traceback

        traceback.print_exc()
        yield f"Error: {str(e)}"


async def query_document(
    question: str, doc_path: str, test: bool = False, session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Document-based research and answer generation (Async Streaming) with conversation memory.

    Args:
        question: User's question
        doc_path: Path to the document
        test: Whether to run in test mode
        session_id: Optional session ID for conversation memory
    """
    if test:
        yield "Test document research response"
        return

    print(f"--- Querying document (Async): {doc_path} | Session: {session_id} ---")

    try:
        # Store user message in memory before querying
        if session_id:
            memory_manager.add_message(session_id, "user", question)

        reader = SimpleDirectoryReader(input_files=[doc_path])
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)

        # Build query with conversation context
        conversation_context = ""
        if session_id:
            history_str = memory_manager.get_history_as_string(session_id)
            if history_str:
                conversation_context = f"\n\nPrevious conversation:\n{history_str}\n\n"

        enhanced_question = (
            f"{conversation_context}Current question: {question}"
            if conversation_context
            else question
        )

        query_engine = index.as_query_engine(
            response_mode="compact", similarity_top_k=3, streaming=True
        )

        streaming_response = await query_engine.aquery(enhanced_question)

        full_response = ""
        async for text in streaming_response.async_response_gen():
            full_response += text
            yield text
            await asyncio.sleep(0)

        # Store assistant response in memory
        if session_id and full_response:
            memory_manager.add_message(session_id, "assistant", full_response)

    except Exception as e:
        print(f"Error in query_document: {e}")
        yield f"Error: {str(e)}"
