"""
Portfolio Allocation Skill

A tool for generating optimized cryptocurrency portfolio allocations using Modern Portfolio Theory. The tool analyzes market sentiment 
using multiple technical indicators from TAapi and Gate.io exchange data, providing detailed insights based on Sharpe ratios, expected 
returns, and volatility. It supports customizable risk tolerance levels and investment amounts while offering AI-powered recommendations 
through GPT-4 for portfolio optimization. The tool can maximize Sharpe ratios or adjust allocations based on user risk preferences, 
while providing comprehensive analysis of market conditions and potential portfolio adjustments.
"""

# Standard library imports
import json
import os
from datetime import datetime
from typing import Dict, Optional, List, Any

# Third party imports
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from openai import OpenAI
from scipy.optimize import minimize


class PortfolioAllocation:
    """A class for optimizing cryptocurrency portfolio allocations using Modern Portfolio Theory.

    This class provides methods to analyze and optimize cryptocurrency portfolios based on
    historical price data and user preferences. It uses Modern Portfolio Theory to determine
    optimal asset weights while considering risk tolerance levels. The class integrates with
    Gate.io exchange for price data, TAapi for technical indicators, and OpenAI for natural
    language processing of user requests.
    """


    def __init__(self):
        """Initialize the PortfolioAllocation class.

        Sets up API clients and configuration by loading required API keys from environment variables.
        Raises ValueError if required API keys are not found.

        Required environment variables:
            TAAPI_API_KEY: API key for TAapi service to fetch technical indicators
            OPENAI_API_KEY: API key for OpenAI services used for natural language processing
        """
        # Load environment variables
        load_dotenv()
        
        self.taapi_api_key = os.getenv("TAAPI_API_KEY")
        if not self.taapi_api_key:
            raise ValueError("TAAPI_API_KEY environment variable is not set")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.taapi_base_url = "https://api.taapi.io"


    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        try:
            tokens, risk, amount, currency = self.parse_input(prompt)
            if tokens == []:
                return {"response": "No tokens provided. Please provide a list of tokens to analyze.", "metadata": {}}

            token_prices = self.get_token_prices(tokens)
            if token_prices is None:
                return {"response": "No token prices found. Please try again.", "metadata": {}}

            portfolio_weights = self.allocate_portfolio(token_prices, risk)
            allocation_amounts = self.get_allocation_amounts(portfolio_weights, amount, currency)

            annualized_returns, annualized_volatility, sharpe_ratios = self.portfolio_stats(token_prices)
            sentiment = self.get_market_sentiment()

            insights = self.generate_insights(prompt, allocation_amounts, annualized_returns, annualized_volatility, sharpe_ratios, sentiment, risk)

            # Store all context in metadata
            metadata = {
                "allocation_amounts": allocation_amounts,
                "prompt": prompt,
                "exchange": "gateio",
                "tokens": tokens,
                "risk": risk,
                "amount": amount,
                "currency": currency,
                "portfolio_weights": portfolio_weights,
                "annualized_returns": annualized_returns,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratios": sharpe_ratios,
                "market_sentiment": sentiment,
                "timestamp": datetime.now().isoformat()
            }

            return {"response": insights, "metadata": metadata}

        except Exception as e:
            return {"error": str(e)}
        

    def parse_input(self, prompt: str) -> tuple[List[str], str, float, Optional[str]]:
        """Parse user input prompt to extract portfolio parameters.

        Uses GPT-4 to analyze the natural language prompt and extract:
        - List of cryptocurrency token symbols
        - Risk tolerance level (low/medium/high/none) 
        - Investment amount (if specified)
        - Currency symbol/code (if amount specified)

        Args:
            prompt: Natural language string describing desired portfolio

        Returns:
            Tuple containing:
            - List of token symbols (e.g. ["BTC", "ETH"])
            - Risk tolerance string ("low", "medium", "high", or "none")
            - Investment amount as float (0.0 if not specified)
            - Currency symbol/code (None if not specified)

        Raises:
            HTTPException: If there is an error parsing the prompt
        """
        context = f"""Extract cryptocurrency token symbols, risk tolerance, and investment amount from the following request.
Risk tolerance levels are: low, medium, high, none

Example inputs and outputs:
Input: "analyze BTC, ETH and SOL with high risk"
Output: {{"tokens": ["BTC", "ETH", "SOL"], "risk": "high", "amount": 0, "currency": null}}

Input: "what do you think about Bitcoin and Ethereum with $2000 investment"
Output: {{"tokens": ["BTC", "ETH"], "risk": "none", "amount": 2000, "currency": "$"}}

Input: "conservative portfolio with USDT and BTC investing 500 japanese yen"
Output: {{"tokens": ["USDT", "BTC"], "risk": "low", "amount": 500, "currency": "JPY"}}

Now extract from this request: "{prompt}"

IMPORTANT: Respond with ONLY the raw JSON object. Do not include markdown formatting, code blocks, or any other text. The response should start with {{ and end with }}."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading expert that extracts token symbols, risk tolerance, and investment amounts from portfolio requests. Always respond with a valid JSON object.",
                    },
                    {"role": "user", "content": context},
                ],
                temperature=0,
            )

            response_text = response.choices[0].message.content.strip()

            try:
                data = json.loads(response_text)
                return (
                    data.get("tokens", []),
                    data.get("risk", "none"),
                    data.get("amount", 0.0),
                    data.get("currency", None)
                )
            except json.JSONDecodeError:
                return [], "none", 0.0, None

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error parsing portfolio request: {str(e)}"
            )


    def get_available_symbols(self, exchange: str) -> List[str]:
        """Fetch available trading pairs from an exchange.

        Retrieves the list of trading pairs from the specified exchange via the TAapi service.
        Filters for USDT pairs and formats them consistently. Falls back to a default list
        if the API request fails.

        Args:
            exchange: Name of exchange to fetch symbols from (e.g. 'gateio')

        Returns:
            List of trading pairs in 'TOKEN/USDT' format (e.g. ['BTC/USDT', 'ETH/USDT'])
            If API fails, returns fallback list of major pairs
        """
        try:
            url = f"{self.taapi_base_url}/exchange-symbols"
            response = requests.get(
                url, params={"secret": self.taapi_api_key, "exchange": exchange}
            )

            if not response.ok:
                print(f"\nError fetching symbols: {response.status_code}")
                print(f"Response: {response.text}")
                return self._get_fallback_symbols()

            symbols = response.json()
            if not symbols or not isinstance(symbols, list):
                print("\nInvalid response format from symbols endpoint")
                return self._get_fallback_symbols()

            # Filter for USDT pairs and ensure proper formatting
            formatted_pairs = [
                symbol
                for symbol in symbols
                if isinstance(symbol, str) and symbol.endswith("/USDT")
            ]

            if formatted_pairs:
                return sorted(formatted_pairs)

            return self._get_fallback_symbols()

        except Exception as e:
            print(f"\nError fetching trading pairs: {str(e)}")
            return self._get_fallback_symbols()


    def find_pair(self, token: str, symbols: List[str]) -> Optional[str]:
        """Find the matching USDT trading pair for a given token.

        Takes a token symbol and list of available trading pairs, attempts to find the 
        corresponding USDT trading pair by doing an exact match. The token is standardized
        by converting to uppercase and removing any '/USDT' suffix if present.

        Args:
            token: Token symbol to find pair for (e.g. 'BTC', 'ETH')
            symbols: List of available trading pairs in format 'TOKEN/USDT'

        Returns:
            Optional[str]: Matching trading pair if found (e.g. 'BTC/USDT'), 
                         None if no match found
        """
        try:
            # Clean and standardize token
            token = token.strip().upper()
            # Remove /USDT if present
            token = token.replace('/USDT', '')
            
            # Try exact USDT pair match
            exact_match = f"{token}/USDT"
            if exact_match in symbols:
                return exact_match
            return None

        except Exception as e:
            print(f"\nError finding best pair: {str(e)}")
            return None


    def get_token_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch historical price data for a trading pair from TAapi.
        
        Makes a request to TAapi's candle endpoint to get 300 days of historical 
        daily price data for the given trading pair on Gate.io exchange.

        Args:
            symbol: Trading pair symbol in format 'TOKEN/USDT'

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing candle data if successful,
                                    None if request fails

        Raises:
            HTTPException: If there is an error fetching the price data
        """
        try:
            payload = {
                "secret": self.taapi_api_key,
                "exchange": "gateio",
                "symbol": symbol,
                "interval": "1d",
                "results": 300  # Get 300 days of historical data
            }

            url = f"{self.taapi_base_url}/candle"  # Use candles endpoint instead of price
            response = requests.get(url = url, params = payload)

            if not response.ok:
                print(f"Error Response Status: {response.status_code}")
                print(f"Error Response Content: {response.text}")
                return None

            return response.json()

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching price data: {str(e)}"
            )
        

    def get_token_prices(self, tokens: List[str]) -> pd.DataFrame:
        """Fetch historical price data for a list of tokens from the Gate.io exchange.

        Retrieves daily closing prices for each token by finding its USDT trading pair
        and fetching historical data. Combines prices into a DataFrame with tokens as columns
        and timestamps as index.

        Args:
            tokens: List of token symbols to fetch prices for (e.g. ["BTC", "ETH"])

        Returns:
            DataFrame containing historical daily closing prices for each token,
            with token symbols as column names and timestamps as index
        """
        # Get all available symbols from Gate.io
        gateio_symbols = self.get_available_symbols("gateio")
        # Find the best trading pair for each token
        pairs = [self.find_pair(token, gateio_symbols) for token in tokens]

        if pairs is None:
            return None
        
        token_prices = {}
        for pair in pairs:
            if pair is not None:
                data = self.get_token_data(pair)
                if data:
                    # Extract prices to dataframe
                    prices = pd.DataFrame({"timestamp": data["timestamp"], "close": data["close"]})
                    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="s")
                    prices.set_index("timestamp", inplace=True)
                    token_prices[pair] = prices["close"]

        df = pd.concat(token_prices, axis=1)
        df.columns = pairs
        return df
    
    
    def allocate_portfolio(self, token_prices: pd.DataFrame, risk_tolerance: str) -> Dict[str, float]:
        """Determine optimal portfolio allocation weights using Modern Portfolio Theory.

        Optimizes portfolio weights based on historical prices and risk tolerance level.
        For risk_tolerance="none", maximizes the Sharpe ratio.
        For other risk levels, optimizes return vs volatility with risk-adjusted constraints.

        Args:
            token_prices: DataFrame of historical token prices with tokens as columns
            risk_tolerance: Risk preference ("low", "medium", "high", or "none")

        Returns:
            Dictionary mapping token symbols to their optimal portfolio weights
        """
        # Calculate daily returns
        returns = token_prices.pct_change().dropna()

        # Average returns and covariance matrix
        expected_returns = returns.mean()  
        cov_matrix = returns.cov()

        def negative_sharpe(weights, returns, cov_matrix):
            portfolio_return = np.sum(returns * weights)
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = (portfolio_return) / portfolio_volatility
            return -sharpe_ratio
        
        # Objective function (maximize return, adjusted for risk)
        def objective(weights, returns, cov_matrix, risk_aversion=1.0):
            portfolio_return = np.sum(returns * weights)
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            # Utility function: return - risk_aversion * volatility^2
            return -(portfolio_return - risk_aversion * portfolio_volatility**2)
        
        n_assets = len(expected_returns)

        # Constraints: Weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial equal weights
        initial_weights = np.array([1/n_assets] * n_assets)

        # Risk-specific settings
        if risk_tolerance == "low":
            max_volatility = 0.03  # 3% daily
            risk_aversion = 5.0    # High penalty on volatility
            constraints.append({"type": "ineq", "fun": lambda w: max_volatility - np.sqrt(w.T @ cov_matrix @ w)})
        elif risk_tolerance == "medium":
            max_volatility = 0.06  # 6% daily
            risk_aversion = 2.0    # Moderate penalty
            constraints.append({"type": "ineq", "fun": lambda w: max_volatility - np.sqrt(w.T @ cov_matrix @ w)})
        else:  # high risk
            risk_aversion = 0.5    # Low penalty

        if risk_tolerance == "none":
            # Maximize Sharpe Ratio(minimize negative)
            optimal_result = minimize(negative_sharpe, initial_weights, args=(expected_returns, cov_matrix), 
                                    method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            optimal_result = minimize(objective, initial_weights, args=(expected_returns, cov_matrix, risk_aversion), 
                                    method="SLSQP", bounds=bounds, constraints=constraints)

        optimal_weights = optimal_result.x

        # Convert optimal weights to dictionary
        portfolio_weights = {}
        for pair, weight in zip(token_prices.keys(), optimal_weights):
            token = pair.replace('/USDT', '')  # Remove /USDT suffix
            portfolio_weights[token] = float(weight)  # Convert numpy float to Python float

        # Round weights to 2 decimal places
        rounded_weights = {token: round(weight, 2) for token, weight in portfolio_weights.items()}
        
        # Calculate rounding error and adjust largest weight to ensure sum is 1.0
        weight_sum = sum(rounded_weights.values())
        if weight_sum != 1.0:
            # Find token with largest weight
            max_token = max(rounded_weights, key=rounded_weights.get)
            # Adjust its weight to make sum equal 1.0
            rounded_weights[max_token] = round(rounded_weights[max_token] + (1.0 - weight_sum), 2)
            
        portfolio_weights = rounded_weights
        
        return portfolio_weights
    

    def get_allocation_amounts(self, portfolio: Dict[str, float], amount: float, currency: Optional[str] = None) -> Dict[str, str]:
        """Convert portfolio weights to currency amounts or percentages.

        Takes a portfolio allocation dictionary with token weights and converts them to either:
        1. Currency amounts if an investment amount is provided (e.g. "$1,000.00")
        2. Percentage strings if no amount provided (e.g. "25%")

        Args:
            portfolio: Dictionary mapping tokens to their portfolio weights (0-1)
            amount: Total investment amount in currency units
            currency: Optional currency symbol/code to prepend to amounts

        Returns:
            Dictionary mapping tokens to formatted allocation strings
        """
        allocation_amounts = {}

        if amount != 0.0:
            # First pass - round all amounts to 2 decimals
            allocated_amounts = {}
            for token, weight in portfolio.items():
                allocated_amounts[token] = round(amount * weight, 2)
            
            # Calculate rounding error and adjust largest allocation
            total_allocated = sum(allocated_amounts.values())
            if total_allocated != amount:
                # Find token with largest allocation
                max_token = max(allocated_amounts, key=allocated_amounts.get)
                # Adjust its allocation to make sum equal amount
                allocated_amounts[max_token] = round(allocated_amounts[max_token] + (amount - total_allocated), 2)
            
            # Format amounts with currency
            for token, allocated_amount in allocated_amounts.items():
                # Format number with commas and 2 decimal places
                formatted_amount = "{:,.2f}".format(allocated_amount)
                
                if currency is not None:
                    if len(currency) > 1:  # Currency code like JPY, EUR
                        allocation_amounts[token] = currency + ' ' + formatted_amount
                    else:  # Currency symbol like $, €
                        allocation_amounts[token] = currency + formatted_amount
                else:
                    allocation_amounts[token] = formatted_amount
        else:
            # Format percentages
            for token, weight in portfolio.items():
                allocation_amounts[token] = str(weight * 100) + '%'

        return allocation_amounts
    

    def portfolio_stats(self, token_prices: pd.DataFrame) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Calculate key portfolio statistics from historical token prices.

        Computes annualized returns, volatility and Sharpe ratios for each token based on daily price data.
        Returns are annualized assuming 365 trading days per year. All metrics are formatted as strings
        with returns and volatility as percentages and Sharpe ratios as decimals.

        Args:
            token_prices: DataFrame containing historical daily prices for each token

        Returns:
            tuple containing:
            - Dict mapping tokens to annualized returns as percentage strings
            - Dict mapping tokens to annualized volatility as percentage strings  
            - Dict mapping tokens to Sharpe ratios as decimal strings
        """
        # Remove '/USDT' from column names
        token_prices.columns = [col.replace('/USDT', '') for col in token_prices.columns]
        trading_days = 365

        # Calculate daily returns, annualized returns, and annualized volatility. As well as Sharpe ratios.
        daily_returns = token_prices.pct_change().dropna()
        annualized_returns = daily_returns.mean() * trading_days
        annualized_volatility = daily_returns.std() * np.sqrt(trading_days)
        sharpe_ratios = (annualized_returns) / annualized_volatility

        # Convert to percentages
        annualized_returns*=100
        annualized_volatility*=100

        # Format the results
        annualized_returns = {k: f"{v:.2f}%" for k,v in annualized_returns.to_dict().items()}
        annualized_volatility = {k: f"{v:.2f}%" for k,v in annualized_volatility.to_dict().items()}
        sharpe_ratios = {k: f"{v:.2f}" for k,v in sharpe_ratios.to_dict().items()}

        return annualized_returns, annualized_volatility, sharpe_ratios
    

    def get_market_sentiment(self) -> int:
        """
        Calculates overall market sentiment based on multiple technical indicators for BTC/USDT.
        
        Uses a combination of indicators including EMA, SMA, RSI, MACD, Bollinger Bands, ROC, 
        Stochastic, VWAP and ADX from the TAapi service to determine if the market is bullish,
        bearish or neutral.

        Returns:
            int: A sentiment score where:
                 Positive values indicate bullish sentiment
                 Negative values indicate bearish sentiment 
                 Zero indicates neutral sentiment
        """
        url = f"{self.taapi_base_url}/bulk"
        
        indicators = [
            {"indicator": 'price'},
            {"indicator": "ema", "period": 12},
            {"indicator": "sma", "period": 50},
            {"indicator": "sma", "period": 200},
            {"indicator": "rsi"},
            {"indicator": "macd"},
            {"indicator": "bbands"},
            {"indicator": "roc"},
            {"indicator": "stoch"},
            {"indicator": "vwap"},
            {"indicator": "adx"}
        ]

        payload = {
            "secret": self.taapi_api_key,
            "construct": {
                "exchange": "gateio",
                "symbol": "BTC/USDT",
                "interval": "1d",
                "indicators": indicators
            }
        }
            
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error fetching market sentiment: {response.text}")

        response_data = response.json()

        # Extract the results for each indicator
        indicator_results = {}
        for indicator in response_data["data"]:
            if indicator["indicator"] == "sma":
                if "sma_50" in indicator["id"]:
                    indicator_results["sma_50"] = indicator["result"]
                elif "sma_200" in indicator["id"]:
                    indicator_results["sma_200"] = indicator["result"]
            else:
                indicator_results[indicator["indicator"]] = indicator["result"]

        sentiment = 0 #Neutral

        # Sentiment is the sum of all the indicators below

        #EMA
        if indicator_results["ema"]["value"] < indicator_results["price"]["value"]:
            sentiment+=1 #Bullish
        elif indicator_results["ema"]["value"] > indicator_results["price"]["value"]:
            sentiment-=1 #Bearish

        #SMA
        if indicator_results["sma_50"]["value"] > indicator_results["sma_200"]["value"]:
            sentiment+=1 #Bullish
        elif indicator_results["sma_50"]["value"] < indicator_results["sma_200"]["value"]:
            sentiment-=1 #Bearish

        #RSI
        if indicator_results["rsi"]["value"] < 30:
            sentiment+=1 #Bullish
        elif indicator_results["rsi"]["value"] < 70:
            sentiment+=0 #Neutral
        else:
            sentiment-=1 #Bearish

        #MACD
        if indicator_results["macd"]["valueMACD"] > indicator_results["macd"]["valueMACDSignal"]:
            sentiment+=1 #Bullish
        elif indicator_results["macd"]["valueMACD"] < indicator_results["macd"]["valueMACDSignal"]:
            sentiment-=1 #Bearish

        #BBANDS
        if indicator_results["price"]["value"] < indicator_results["bbands"]["valueLowerBand"]:
            sentiment+=1 #Bullish
        elif indicator_results["price"]["value"] > indicator_results["bbands"]["valueUpperBand"]:
            sentiment-=1 #Bearish
            
        #ROC
        if indicator_results["roc"]["value"] > 0.5:
            sentiment+=1 #Bullish
        elif indicator_results["roc"]["value"] < -0.5:
            sentiment-=1 #Bearish
            
        #Stoch
        if indicator_results["stoch"]["valueK"] < 20:
            sentiment+=1 #Bullish
        elif indicator_results["stoch"]["valueK"] > 80:
            sentiment-=1 #Bearish

        #VWAP
        if indicator_results["vwap"]["value"] < indicator_results["price"]["value"]:
            sentiment+=1 #Bullish
        elif indicator_results["vwap"]["value"] > indicator_results["price"]["value"]:
            sentiment-=1 #Bearish

        #ADX
        if indicator_results["adx"]["value"] > 25:
            sentiment+=1 #Bullish
        elif indicator_results["adx"]["value"] < 20:
            sentiment-=1 #Bearish
        
        return sentiment
    

    def generate_insights(
        self,
        prompt: str,
        allocation_amounts: Dict[str, str],
        expected_returns: Dict[str, str], 
        volatility: Dict[str, str],
        sharpe_ratios: Dict[str, str],
        sentiment: int,
        risk: str
    ) -> Dict[str, Any]:
        """Generate portfolio insights and recommendations using LLM analysis.
        
        Args:
            allocation_amounts: Dictionary of token allocations and amounts
            sharpe_ratios: Dictionary of Sharpe ratios per token
            expected_returns: Dictionary of expected returns per token
            sentiment: Overall market sentiment score
            risk: User's risk tolerance level
            
        Returns:
            Dictionary containing insights and recommendations
        """
        # Format the data for the LLM prompt
        sentiment_desc = "bullish" if sentiment > 0 else "bearish" if sentiment < 0 else "neutral"
        if risk == "none":
            risk = "Maximize Sharpe Ratio"
        
        context = f"""Analyze this cryptocurrency portfolio and provide detailed insights:

        Portfolio Allocation:
        {json.dumps(allocation_amounts, indent=2)}
        
        Performance Metrics:
        - Sharpe Ratios: {json.dumps(sharpe_ratios, indent=2)}
        - Expected Returns: {json.dumps(expected_returns, indent=2)}
        - Volatility: {json.dumps(volatility, indent=2)}
        
        Market Context:
        - Overall Market Sentiment: {sentiment_desc} ({sentiment})
        - User Risk Tolerance: {risk}

        Please provide:
        1. Analysis of the current allocation strategy based on the user's risk tolerance and market sentiment.
        2. Key risks and opportunities
        3. Specific recommendations for portfolio optimization based on the user's risk tolerance.
        4. Suggestions for token replacements(Example: "Consider replacing XRP with BTC") based on Performance Metrics and Market Context.

        End your response with "Thank you for using the Portfolio Allocation Skill!"
        Note that the Market Context isn't being provided by the User. It is being generated by the tool based on the market data.
        It is not necessary to provide sections 3 and 4 if there are no recommendations to make. Any recommendations need to be significant.
        Note that if the risk tolerance isn't 'low', 'medium', or 'high', the portfolio allocation is maximizing the Sharpe Ratio.
        Also note that the portfolio allocation is using Modern Portfolio Theory. Assume that the portfolio allocation is already optimized based
        on the user's risk tolerance and that it will be the preferred portfolio. Thus, any recommendations are simply alternatives to this portfolio.
        If the user's risk tolerance is 'Maximize Sharpe Ratio', then the portfolio allocation is maximizing the Sharpe Ratio and this must be considered
        in place of the user's risk tolerance.

        To further help you, here is the user's request:
        {prompt}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cryptocurrency portfolio analyst providing detailed insights and recommendations."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=750
        )

        insights = (
            "Here is your suggested portfolio allocation:\n\n"
            + json.dumps(allocation_amounts, indent=2)
            + "\n\nHere are additional insights to help you further optimize your portfolio:\n\n"
            + response.choices[0].message.content
        )

        return insights


def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return PortfolioAllocation().run(prompt, system_prompt)