# 📈 AI-Powered Stock Trading Agent

A sophisticated multi-agent AI system for comprehensive stock analysis that combines technical indicators, sentiment analysis from multiple sources, and risk assessment to provide actionable trading recommendations.

## 🌟 Features

### Multi-Agent Architecture
- **Technical Analysis Agent**: Analyzes price data using multiple indicators (RSI, Alligator, Squeeze Momentum)
- **Sentiment Analysis Agent**: Aggregates sentiment from Twitter, Reddit, AskNews, and Yahoo Finance
- **Risk Assessment Agent**: Evaluates investment risks and recommends position sizing
- **Portfolio Recommendation Agent**: Synthesizes all analyses into actionable trading decisions
- **Chat Agent**: Provides conversational interface for follow-up questions and analysis clarification

### Technical Indicators
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **Alligator Indicator**: Detects trend direction and strength
- **Squeeze Momentum**: Identifies volatility compression and potential breakouts
- Voting-based consensus system with confidence weighting

### Sentiment Analysis Sources
- **Twitter/X**: Social media sentiment tracking
- **Reddit**: Community discussions from finance subreddits
- **AskNews**: Professional news sentiment analysis
- **Yahoo Finance**: Financial news headlines
- Multi-source aggregation with weighted scoring

### Risk Management
- Position size recommendations based on risk tolerance
- Volatility assessment
- Key risk identification
- Risk mitigation strategies
- Stop-loss and take-profit level calculations

### LLM Provider Support
- **Anthropic Claude** (Haiku, Sonnet, Opus)
- **OpenAI** (GPT-4, GPT-3.5)
- **Google Gemini**
- **Ollama** (Local models like Qwen3)

### User Interface
- **Gradio Web Interface** with three main tabs:
  - Stock Analysis: Run comprehensive analyses
  - Chat & Follow-up: Ask questions about analyses
  - History: View session history and past analyses
- **Programmatic API** for automated trading systems

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- Internet connection for API calls

### API Keys Required
- **LangSmith** (optional, for tracing)
- **LLM Provider**: At least one of:
  - Anthropic API key
  - OpenAI API key
  - Google API key
  - Ollama (local, no API key needed)
- **Financial Data**:
  - Polygon API key (optional)
  - Tavily API key (for web search)
- **Social Media**:
  - Twitter API credentials (optional)
  - Reddit API credentials (optional)
- **News**:
  - AskNews API credentials (optional)

## 🚀 Installation

### 1. Clone or Download
```bash
# If you have the file
# Just ensure financ_agent.py is in your working directory
```

### 2. Install Dependencies
```bash
pip install --upgrade langchain langchain-core langchain-community langchain-experimental
pip install langchain_openai langchain-google-genai langchain-ollama langchain-anthropic langchain-tavily
pip install praw yfinance asknews gradio sqlite3 tweepy
pip install python-dotenv pydantic numpy
pip install langgraph langsmith
```

### 3. Set Up Environment Variables
Create a `.env` file in the same directory:

```env
# LangSmith (Optional - for tracing)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=stock-trading-agent
LANGSMITH_TRACING_V2=true

# LLM Provider (Choose at least one)
LLM_PROVIDER=anthropic  # Options: anthropic, openai, gemini, ollama

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL_NAME=claude-haiku-4-5-20251001

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4-turbo-preview

# Google
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-3-pro-preview

# Ollama (local)
OLLAMA_MODEL=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434

# Financial Data APIs
POLYGON_API_KEY=your_polygon_api_key
TAVILY_API_KEY=your_tavily_api_key

# AskNews
ASKNEWS_CLIENT_ID=your_asknews_client_id
ASKNEWS_CLIENT_SECRET=your_asknews_client_secret

# Twitter/X API (Optional)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Reddit API (Optional)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=StockTradingAgent/1.0

# Configuration
MAX_TWEETS=50
MAX_REDDIT_POSTS=50
SOCIAL_LOOKBACK_DAYS=7
MODEL_TEMPERATURE=0
DEFAULT_RISK_TOLERANCE=medium
MAX_POSITION_SIZE=0.15
MIN_CONFIDENCE_THRESHOLD=60
DB_PATH=trading_history.db
```

### 4. Set Up Ollama (Optional - for local models)
If using Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# Pull the model
ollama pull qwen3:latest
```

## 💻 Usage

### Option 1: Web Interface (Recommended)

```python
python financ_agent.py
```

This launches a Gradio interface where you can:
1. Enter a stock ticker (e.g., NVDA, AAPL, TSLA)
2. Select risk tolerance (low, medium, high, very_high)
3. Run analysis and view comprehensive reports
4. Chat with the AI about your analysis
5. View session history

### Option 2: Programmatic Usage

Edit the file to set `RUN_WITH_UI = False` and customize the ticker:

```python
if __name__ == "__main__":
    ticker = "NVDA"
    
    result = analyze_stock(
        ticker=ticker,
        risk_tolerance="medium",
        save_to_file=True
    )
    
    print_analysis_report(result)
```

### Option 3: Import as Module

```python
from financ_agent import analyze_stock, print_analysis_report

# Run analysis
result = analyze_stock(
    ticker="TSLA",
    risk_tolerance="high",
    save_to_file=True
)

# Print formatted report
print_analysis_report(result)

# Access specific components
recommendation = result.get('recommendation', {})
decision = recommendation.get('decision')  # BUY, SELL, or HOLD
confidence = recommendation.get('confidence')  # 0-100
```

## 📊 Output Structure

The analysis returns a comprehensive dictionary with:

```python
{
    'ticker': 'NVDA',
    'recommendation': {
        'decision': 'BUY',  # BUY, SELL, or HOLD
        'confidence': 85,
        'decision_rationale': '...',
        'primary_driver': 'technical_bullish_sentiment_positive',
        'entry_price': 145.32,
        'stop_loss': 138.05,
        'take_profit_1': 152.59,
        'take_profit_2': 159.86,
        'position_size': 0.12,
        'alternative_scenarios': {...}
    },
    'technical_analysis': {
        'final_signal': 'BUY',
        'confidence': 78,
        'indicators': [...],
        'raw_data': {...}
    },
    'sentiment_analysis': {
        'overall_sentiment': 'positive',
        'sentiment_score': 0.65,
        'sources': {...}
    },
    'risk_assessment': {
        'risk_level': 'medium',
        'risk_score': 45,
        'key_risks': [...],
        'volatility_assessment': 'medium'
    },
    'timestamp': '2026-01-27T...'
}
```

## 🔧 Configuration Options

### Risk Tolerance Levels
- **low**: Conservative approach, smaller positions
- **medium**: Balanced risk/reward (default)
- **high**: Aggressive approach, larger positions
- **very_high**: Maximum risk tolerance

### Customizable Parameters
Edit `Config` class for:
- `MAX_TWEETS`: Maximum tweets to analyze (default: 50)
- `MAX_REDDIT_POSTS`: Maximum Reddit posts (default: 50)
- `SOCIAL_LOOKBACK_DAYS`: Days to look back for social data (default: 7)
- `MODEL_TEMPERATURE`: LLM creativity level (default: 0)
- `MAX_POSITION_SIZE`: Maximum portfolio allocation (default: 0.15 = 15%)
- `MIN_CONFIDENCE_THRESHOLD`: Minimum confidence for recommendations (default: 60)

## 🗄️ Database & History

The agent stores analysis history in SQLite (`trading_history.db`):
- All stock analyses with timestamps
- Chat conversations
- Session management
- Easy retrieval for backtesting

## 🏗️ Architecture

### Agent Workflow
```
User Query → Orchestrator
    ↓
Technical Analysis Agent (parallel)
    - Fetches price data
    - Calculates indicators
    - Votes on signals
    ↓
Sentiment Analysis Agent (parallel)
    - Twitter sentiment
    - Reddit sentiment
    - AskNews sentiment
    - Yahoo Finance news
    ↓
Risk Assessment Agent
    - Evaluates volatility
    - Identifies risks
    - Recommends position size
    ↓
Portfolio Recommendation Agent
    - Synthesizes all data
    - Makes final decision
    - Calculates price targets
    ↓
Chat Agent (for follow-ups)
```

### LangGraph State Management
Uses LangGraph for:
- Stateful agent coordination
- Message passing between agents
- Conversation memory
- Checkpointing for resumability

## 🛠️ Troubleshooting

### Common Issues

**1. "No module named 'langchain'"**
```bash
pip install --upgrade langchain langchain-core
```

**2. "API key not found"**
- Ensure `.env` file is in the same directory
- Check environment variable names match exactly

**3. "Ollama connection refused"**
```bash
ollama serve  # Start the Ollama server
```

**4. "Rate limit exceeded"**
- Reduce `MAX_TWEETS` and `MAX_REDDIT_POSTS`
- Add delays between API calls
- Check API rate limits for your tier

**5. "Twitter/Reddit API not working"**
- These are optional; the agent will skip if unavailable
- Check API credentials are correct
- Ensure proper app permissions

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- **Not Financial Advice**: This agent provides analysis, not investment recommendations
- **No Guarantees**: Past performance doesn't indicate future results
- **Use at Your Own Risk**: Always do your own research
- **API Costs**: Be aware of API usage costs from various providers
- **Market Volatility**: Markets can change rapidly; analyses may be outdated quickly

## 🔐 Security Best Practices

1. **Never commit `.env` files** to version control
2. **Rotate API keys** regularly
3. **Use read-only API keys** where possible
4. **Monitor API usage** to prevent unexpected charges
5. **Keep dependencies updated** for security patches

## 📝 Logging & Monitoring

### LangSmith Integration
When enabled, LangSmith provides:
- Complete trace of agent interactions
- Performance metrics
- Debugging capabilities
- Cost tracking per run

### Console Logging
The agent logs:
- Analysis progress
- API calls
- Errors and warnings
- Execution time

## 🤝 Contributing

To extend the agent:
1. Add new tools in the tools section
2. Create new agents by extending the agent structure
3. Modify state schema for additional data
4. Update the orchestrator workflow

## 📄 License

This project is provided as-is for educational purposes. Check individual API providers for their terms of service.

## 🆘 Support

For issues:
1. Check the troubleshooting section
2. Verify all API keys are correctly configured
3. Ensure all dependencies are installed
4. Check API service status pages

## 🎯 Roadmap

Potential enhancements:
- [ ] Backtesting framework
- [ ] More technical indicators
- [ ] Crypto support
- [ ] Portfolio optimization
- [ ] Real-time alerts
- [ ] Mobile app interface
- [ ] Multi-asset correlation analysis
- [ ] Machine learning price predictions

## 📚 References

- **LangChain**: https://python.langchain.com/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Gradio**: https://gradio.app/
- **yfinance**: https://github.com/ranaroussi/yfinance
- **Technical Indicators**: Investopedia.com

---

**Built with ❤️ using LangChain, LangGraph, and various AI providers**