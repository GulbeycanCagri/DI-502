"""
Unit tests for rag_service_2.py
Tests the enhanced financial query processing and data fetching functions.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ===========================================
# Query Intent Analysis Tests
# ===========================================

class TestQueryIntentAnalysis:
    """Tests for analyze_query_intent and related functions."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        # Import after path setup
        from backend.src.rag_service import (
            QueryType,
            QueryIntent,
            analyze_query_intent,
            extract_entities_rule_based,
            TICKER_MAPPING,
            CRYPTO_MAPPING,
        )
        self.QueryType = QueryType
        self.QueryIntent = QueryIntent
        self.analyze_query_intent = analyze_query_intent
        self.extract_entities_rule_based = extract_entities_rule_based
        self.TICKER_MAPPING = TICKER_MAPPING
        self.CRYPTO_MAPPING = CRYPTO_MAPPING
    
    def test_extract_entities_nvidia(self):
        """Test entity extraction for NVIDIA."""
        ticker, crypto, company = self.extract_entities_rule_based("what is nvidia stock price")
        assert ticker == "NVDA"
        assert crypto is None
    
    def test_extract_entities_bitcoin(self):
        """Test entity extraction for Bitcoin."""
        ticker, crypto, company = self.extract_entities_rule_based("bitcoin price today")
        assert ticker is None
        assert crypto == "BTC"
    
    def test_extract_entities_multiword_company(self):
        """Test entity extraction for multi-word company names."""
        ticker, crypto, company = self.extract_entities_rule_based("bank of america news")
        assert ticker == "BAC"
    
    def test_extract_entities_no_match(self):
        """Test entity extraction returns None for unknown entities."""
        ticker, crypto, company = self.extract_entities_rule_based("random company xyz")
        assert ticker is None
        assert crypto is None
    
    @patch('backend.src.rag_service_2.llm')
    def test_analyze_intent_price_lookup(self, mock_llm):
        """Test intent analysis for direct price queries."""
        mock_llm.complete.return_value = '{"intent": "price_lookup", "ticker": "NVDA", "crypto": null, "company": "NVIDIA", "needs_price": true, "needs_news": false, "is_prediction": false}'
        
        intent = self.analyze_query_intent("What is NVIDIA stock price?")
        
        assert intent.query_type == self.QueryType.PRICE_LOOKUP
        assert intent.ticker == "NVDA"
        assert intent.needs_price_data == True
        assert intent.is_prediction_question == False
    
    @patch('backend.src.rag_service_2.llm')
    def test_analyze_intent_price_analysis(self, mock_llm):
        """Test intent analysis for price analysis/prediction queries."""
        mock_llm.complete.return_value = '{"intent": "price_analysis", "ticker": null, "crypto": "BTC", "company": null, "needs_price": true, "needs_news": true, "is_prediction": true}'
        
        intent = self.analyze_query_intent("Do you think Bitcoin price will increase?")
        
        assert intent.query_type == self.QueryType.PRICE_ANALYSIS
        assert intent.crypto_symbol == "BTC"
        assert intent.needs_price_data == True
        assert intent.needs_news == True
        assert intent.is_prediction_question == True
    
    @patch('backend.src.rag_service_2.llm')
    def test_analyze_intent_news_query(self, mock_llm):
        """Test intent analysis for news queries."""
        mock_llm.complete.return_value = '{"intent": "news_query", "ticker": "TSLA", "crypto": null, "company": "Tesla", "needs_price": false, "needs_news": true, "is_prediction": false}'
        
        intent = self.analyze_query_intent("What's the latest news about Tesla?")
        
        assert intent.query_type == self.QueryType.NEWS_QUERY
        assert intent.ticker == "TSLA"
        assert intent.needs_news == True
    
    @patch('backend.src.rag_service_2.llm')
    def test_analyze_intent_market_analysis(self, mock_llm):
        """Test intent analysis for market analysis queries."""
        mock_llm.complete.return_value = '{"intent": "market_analysis", "ticker": null, "crypto": null, "company": null, "needs_price": false, "needs_news": true, "is_prediction": false}'
        
        intent = self.analyze_query_intent("How is the stock market performing?")
        
        assert intent.query_type == self.QueryType.MARKET_ANALYSIS
        assert intent.ticker is None
        assert intent.needs_news == True
    
    def test_ticker_mapping_coverage(self):
        """Test that ticker mapping has good coverage."""
        # Major tech companies
        assert "nvidia" in self.TICKER_MAPPING
        assert "apple" in self.TICKER_MAPPING
        assert "microsoft" in self.TICKER_MAPPING
        assert "google" in self.TICKER_MAPPING
        assert "amazon" in self.TICKER_MAPPING
        
        # Financial companies
        assert "jpmorgan" in self.TICKER_MAPPING
        assert "visa" in self.TICKER_MAPPING
        
        # Verify values are valid ticker symbols
        for ticker in self.TICKER_MAPPING.values():
            assert ticker.isupper()
            assert len(ticker) <= 5
    
    def test_crypto_mapping_coverage(self):
        """Test that crypto mapping has good coverage."""
        assert "bitcoin" in self.CRYPTO_MAPPING
        assert "ethereum" in self.CRYPTO_MAPPING
        assert "solana" in self.CRYPTO_MAPPING
        assert "dogecoin" in self.CRYPTO_MAPPING
        
        # Verify values are valid symbols
        for symbol in self.CRYPTO_MAPPING.values():
            assert symbol.isupper()


# ===========================================
# Data Fetching Tests
# ===========================================

class TestDataFetching:
    """Tests for API data fetching functions."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        from backend.src.rag_service import (
            fetch_stock_quote,
            fetch_company_profile,
            fetch_company_news,
            fetch_general_news,
            fetch_crypto_price,
            fetch_crypto_news,
            build_context_from_data,
            QueryIntent,
            QueryType,
        )
        self.fetch_stock_quote = fetch_stock_quote
        self.fetch_company_profile = fetch_company_profile
        self.fetch_company_news = fetch_company_news
        self.fetch_general_news = fetch_general_news
        self.fetch_crypto_price = fetch_crypto_price
        self.fetch_crypto_news = fetch_crypto_news
        self.build_context_from_data = build_context_from_data
        self.QueryIntent = QueryIntent
        self.QueryType = QueryType
    
    @patch('backend.src.rag_service_2.requests.get')
    def test_fetch_stock_quote_success(self, mock_get):
        """Test successful stock quote fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "c": 150.25,  # current price
            "d": 2.50,    # change
            "dp": 1.69,   # percent change
            "h": 152.00,  # high
            "l": 148.00,  # low
            "o": 149.00,  # open
            "pc": 147.75  # previous close
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = self.fetch_stock_quote("NVDA")
        
        assert result is not None
        assert result["current_price"] == 150.25
        assert result["change"] == 2.50
        assert result["percent_change"] == 1.69
    
    @patch('backend.src.rag_service_2.requests.get')
    def test_fetch_stock_quote_failure(self, mock_get):
        """Test stock quote fetch with API error."""
        mock_get.side_effect = Exception("API Error")
        
        result = self.fetch_stock_quote("INVALID")
        
        assert result is None
    
    @patch('backend.src.rag_service_2.requests.get')
    def test_fetch_company_profile_success(self, mock_get):
        """Test successful company profile fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "NVIDIA Corporation",
            "ticker": "NVDA",
            "finnhubIndustry": "Technology",
            "marketCapitalization": 1500000,
            "weburl": "https://nvidia.com",
            "country": "US"
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = self.fetch_company_profile("NVDA")
        
        assert result is not None
        assert result["name"] == "NVIDIA Corporation"
        assert result["industry"] == "Technology"
    
    @patch('backend.src.rag_service_2.requests.get')
    def test_fetch_company_news_success(self, mock_get):
        """Test successful company news fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"headline": "NVIDIA News 1", "summary": "Summary 1", "source": "Reuters"},
            {"headline": "NVIDIA News 2", "summary": "Summary 2", "source": "Bloomberg"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = self.fetch_company_news("NVDA", days=7)
        
        assert len(result) == 2
        assert result[0]["headline"] == "NVIDIA News 1"
    
    @patch('backend.src.rag_service_2.requests.get')
    def test_fetch_crypto_price_success(self, mock_get):
        """Test successful crypto price fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "bitcoin": {
                "usd": 45000.50,
                "usd_24h_change": 2.5,
                "usd_market_cap": 850000000000
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = self.fetch_crypto_price("BTC")
        
        assert result is not None
        assert result["symbol"] == "BTC"
        assert result["price_usd"] == 45000.50
    
    @patch('backend.src.rag_service_2.requests.get')
    def test_fetch_crypto_price_unknown_symbol(self, mock_get):
        """Test crypto price fetch with unknown symbol."""
        result = self.fetch_crypto_price("UNKNOWN")
        
        assert result is None
        mock_get.assert_not_called()
    
    def test_build_context_with_stock_data(self):
        """Test context building with stock quote data."""
        intent = self.QueryIntent(
            query_type=self.QueryType.STOCK_PRICE,
            ticker="NVDA"
        )
        stock_quote = {
            "current_price": 150.25,
            "change": 2.50,
            "percent_change": 1.69,
            "high": 152.00,
            "low": 148.00,
            "open": 149.00,
            "previous_close": 147.75
        }
        
        context = self.build_context_from_data(intent, stock_quote=stock_quote)
        
        assert "NVDA" in context
        assert "$150.25" in context
        assert "1.69%" in context
    
    def test_build_context_with_crypto_data(self):
        """Test context building with crypto price data."""
        intent = self.QueryIntent(
            query_type=self.QueryType.CRYPTO_PRICE,
            crypto_symbol="BTC"
        )
        crypto_price = {
            "symbol": "BTC",
            "price_usd": 45000.50,
            "change_24h": 2.5,
            "market_cap": 850000000000
        }
        
        context = self.build_context_from_data(intent, crypto_price=crypto_price)
        
        assert "BTC" in context
        assert "$45,000.50" in context
    
    def test_build_context_with_news(self):
        """Test context building with news articles."""
        intent = self.QueryIntent(query_type=self.QueryType.GENERAL_FINANCE)
        news_articles = [
            {"headline": "Market rises", "summary": "Stocks up today", "source": "Reuters"},
            {"headline": "Tech gains", "summary": "Tech sector strong", "source": "Bloomberg"},
        ]
        
        context = self.build_context_from_data(intent, news_articles=news_articles)
        
        assert "Market rises" in context
        assert "Reuters" in context
        assert "Tech gains" in context


# ===========================================
# Memory Manager Tests
# ===========================================

class TestMemoryManager:
    """Tests for conversation memory management."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        from backend.src.memory_manager import ConversationMemoryManager
        self.manager = ConversationMemoryManager(
            token_limit=1000,
            session_timeout=60,
            max_sessions=10
        )
    
    def test_create_new_session(self):
        """Test creating a new session."""
        session_id = "test-session-1"
        memory = self.manager.get_or_create_session(session_id)
        
        assert memory is not None
        assert self.manager.session_exists(session_id)
    
    def test_add_and_retrieve_messages(self):
        """Test adding and retrieving messages."""
        session_id = "test-session-2"
        
        self.manager.add_message(session_id, "user", "Hello")
        self.manager.add_message(session_id, "assistant", "Hi there!")
        
        history = self.manager.get_chat_history(session_id)
        
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there!"
    
    def test_get_history_as_string(self):
        """Test getting history as formatted string."""
        session_id = "test-session-3"
        
        self.manager.add_message(session_id, "user", "Question")
        self.manager.add_message(session_id, "assistant", "Answer")
        
        history_str = self.manager.get_history_as_string(session_id)
        
        assert "User: Question" in history_str
        assert "Assistant: Answer" in history_str
    
    def test_clear_session(self):
        """Test clearing a session."""
        session_id = "test-session-4"
        self.manager.add_message(session_id, "user", "Test")
        
        assert self.manager.session_exists(session_id)
        
        result = self.manager.clear_session(session_id)
        
        assert result is True
        assert not self.manager.session_exists(session_id)
    
    def test_clear_nonexistent_session(self):
        """Test clearing a non-existent session returns False."""
        result = self.manager.clear_session("nonexistent")
        
        assert result is False
    
    def test_get_session_count(self):
        """Test getting session count."""
        initial_count = self.manager.get_session_count()
        
        self.manager.add_message("session-a", "user", "A")
        self.manager.add_message("session-b", "user", "B")
        
        assert self.manager.get_session_count() == initial_count + 2


# ===========================================
# Integration Tests (Async)
# ===========================================

class TestAsyncFunctions:
    """Tests for async streaming functions."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        from backend.src.rag_service import plain_chat, query_online, query_document
        self.plain_chat = plain_chat
        self.query_online = query_online
        self.query_document = query_document
    
    @pytest.mark.asyncio
    async def test_plain_chat_test_mode(self):
        """Test plain_chat in test mode."""
        chunks = []
        async for chunk in self.plain_chat("test question", test=True):
            chunks.append(chunk)
        
        result = "".join(chunks)
        assert result == "Test response from Ollama"
    
    @pytest.mark.asyncio
    async def test_query_online_test_mode(self):
        """Test query_online in test mode."""
        chunks = []
        async for chunk in self.query_online("test question", test=True):
            chunks.append(chunk)
        
        result = "".join(chunks)
        assert result == "Test online research response"
    
    @pytest.mark.asyncio
    async def test_query_document_test_mode(self):
        """Test query_document in test mode."""
        chunks = []
        async for chunk in self.query_document("test question", "/fake/path", test=True):
            chunks.append(chunk)
        
        result = "".join(chunks)
        assert result == "Test document research response"


# ===========================================
# Edge Cases and Error Handling Tests
# ===========================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        from backend.src.rag_service import (
            extract_entities_rule_based,
            generate_search_keywords,
        )
        self.extract_entities = extract_entities_rule_based
        self.generate_keywords = generate_search_keywords
    
    def test_extract_entities_case_insensitive(self):
        """Test that entity extraction is case insensitive."""
        ticker1, _, _ = self.extract_entities("NVIDIA stock")
        ticker2, _, _ = self.extract_entities("nvidia stock")
        ticker3, _, _ = self.extract_entities("NvIdIa stock")
        assert ticker1 == "NVDA"
        assert ticker2 == "NVDA"
        assert ticker3 == "NVDA"
    
    def test_extract_entities_with_special_chars(self):
        """Test entity extraction with special characters."""
        ticker, _, _ = self.extract_entities("what about nvidia?")
        assert ticker == "NVDA"
    
    def test_extract_entities_empty_string(self):
        """Test entity extraction with empty string."""
        ticker, crypto, company = self.extract_entities("")
        assert ticker is None
        assert crypto is None
    
    @patch('backend.src.rag_service_2.llm')
    def test_generate_keywords_llm_failure(self, mock_llm):
        """Test keyword generation falls back when LLM fails."""
        mock_llm.complete.side_effect = Exception("LLM Error")
        
        keywords = self.generate_keywords("What is the stock price of Apple?")
        
        # Should return fallback keywords
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    @patch('backend.src.rag_service_2.llm')
    def test_generate_keywords_success(self, mock_llm):
        """Test successful keyword generation."""
        mock_llm.complete.return_value = "Apple, stock, price, investment"
        
        keywords = self.generate_keywords("What is Apple stock price?")
        
        assert isinstance(keywords, list)
        assert "Apple" in keywords
        assert len(keywords) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
