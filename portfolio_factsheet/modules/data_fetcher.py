"""
Data fetcher module for retrieving missing stock prices, exchange rates, and sector data.
Uses yfinance for stock data and exchangerate-api.com for currency exchange rates.
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from functools import lru_cache

from config import (
    YFINANCE_TIMEOUT, EXCHANGERATE_API_URL, EXCHANGERATE_TIMEOUT,
    MAX_API_RETRIES, API_RETRY_DELAY, TICKER_MAPPING,
    SECTOR_CACHE_FILE, FX_CACHE_FILE, PRICE_CACHE_FILE,
    CACHE_ENABLED, CACHE_EXPIRY_DAYS
)

# Set up logging
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of fetched data to reduce API calls."""
    
    def __init__(self):
        self.cache_files = {
            "sector": SECTOR_CACHE_FILE,
            "fx": FX_CACHE_FILE,
            "price": PRICE_CACHE_FILE
        }
        self.cache_data = {}
        self.load_all_caches()
    
    def load_all_caches(self):
        """Load all cache files."""
        for cache_type, cache_file in self.cache_files.items():
            self.load_cache(cache_type)
    
    def load_cache(self, cache_type: str):
        """Load cache from file."""
        cache_file = self.cache_files.get(cache_type)
        if not cache_file:
            logger.error(f"Unknown cache type: {cache_type}")
            return
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache_data[cache_type] = json.load(f)
                logger.info(f"Loaded {cache_type} cache with {len(self.cache_data.get(cache_type, {}))} entries")
            else:
                self.cache_data[cache_type] = {}
                logger.info(f"Created new {cache_type} cache")
        except Exception as e:
            logger.error(f"Error loading {cache_type} cache: {e}")
            self.cache_data[cache_type] = {}
    
    def save_cache(self, cache_type: str):
        """Save cache to file."""
        cache_file = self.cache_files.get(cache_type)
        if not cache_file or cache_type not in self.cache_data:
            return
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data[cache_type], f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {cache_type} cache")
        except Exception as e:
            logger.error(f"Error saving {cache_type} cache: {e}")
    
    def save_all_caches(self):
        """Save all caches."""
        for cache_type in self.cache_data.keys():
            self.save_cache(cache_type)
    
    def get_from_cache(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not CACHE_ENABLED:
            return None
        
        if cache_type not in self.cache_data:
            return None
        
        cache_entry = self.cache_data[cache_type].get(key)
        if not cache_entry:
            return None
        
        # Check if cache entry is expired
        if self._is_cache_expired(cache_entry):
            logger.debug(f"Cache entry expired for {key}")
            del self.cache_data[cache_type][key]
            return None
        
        logger.debug(f"Cache hit for {key}")
        return cache_entry.get("value")
    
    def add_to_cache(self, cache_type: str, key: str, value: Any):
        """Add value to cache."""
        if not CACHE_ENABLED:
            return
        
        if cache_type not in self.cache_data:
            self.cache_data[cache_type] = {}
        
        self.cache_data[cache_type][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "expiry_days": CACHE_EXPIRY_DAYS
        }
        logger.debug(f"Added to {cache_type} cache: {key}")
    
    def _is_cache_expired(self, cache_entry: Dict) -> bool:
        """Check if cache entry is expired."""
        if "timestamp" not in cache_entry:
            return True
        
        try:
            timestamp = datetime.fromisoformat(cache_entry["timestamp"])
            expiry_days = cache_entry.get("expiry_days", CACHE_EXPIRY_DAYS)
            expiry_date = timestamp + timedelta(days=expiry_days)
            
            return datetime.now() > expiry_date
        except Exception:
            return True
    
    def clear_expired_entries(self):
        """Clear expired cache entries."""
        for cache_type in list(self.cache_data.keys()):
            expired_keys = []
            for key, entry in self.cache_data[cache_type].items():
                if self._is_cache_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache_data[cache_type][key]
            
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired entries from {cache_type} cache")


class DataFetcher:
    """Fetches missing data from various APIs."""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PortfolioFactsheetGenerator/1.0'
        })
        self.batch_size = 10  # Number of tickers to fetch in batch
        self.max_concurrent_requests = 3  # Maximum concurrent API requests
    
    def format_ticker_for_yfinance(self, ticker: str) -> Optional[str]:
        """
        Format ticker for yfinance compatibility.
        
        Args:
            ticker: Original ticker symbol
            
        Returns:
            Formatted ticker for yfinance, or None for benchmarks/unsupported
        """
        # Handle benchmarks
        if ticker in ["KOSPI", "S&P", "SPX"]:
            return None  # Benchmarks need special handling
        
        # Handle special cases
        if ticker == "2801 JT":
            return "2801.T"  # Tokyo stock exchange
        
        # Apply suffix mappings
        for suffix, yf_suffix in TICKER_MAPPING.items():
            if ticker.endswith(suffix):
                return ticker.replace(suffix, yf_suffix)
        
        # Default: return as-is (works for most US stocks)
        return ticker
    
    def fetch_stock_price(self, ticker: str, date: datetime) -> Optional[float]:
        """
        Fetch stock price for a specific date using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            date: Date to fetch price for
            
        Returns:
            Stock price or None if not found
        """
        cache_key = f"{ticker}_{date.strftime('%Y-%m-%d')}"
        
        # Try cache first
        cached_price = self.cache_manager.get_from_cache("price", cache_key)
        if cached_price is not None:
            return float(cached_price)
        
        # Handle benchmarks
        if ticker in ["KOSPI", "S&P", "SPX"]:
            return self._fetch_benchmark_price(ticker, date)
        
        formatted_ticker = self.format_ticker_for_yfinance(ticker)
        if formatted_ticker is None:
            logger.warning(f"Cannot fetch price for {ticker} - unsupported ticker format")
            return None
        
        for attempt in range(MAX_API_RETRIES):
            try:
                logger.info(f"Fetching price for {ticker} ({formatted_ticker}) on {date}")
                
                # Download historical data
                stock = yf.Ticker(formatted_ticker)
                
                # Get data for the month containing the date
                start_date = date - timedelta(days=30)
                end_date = date + timedelta(days=1)
                
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    timeout=YFINANCE_TIMEOUT
                )
                
                if hist.empty:
                    logger.warning(f"No price data found for {ticker} around {date}")
                    return None
                
                # Find the closest date to our target date
                hist = hist.sort_index()
                
                # Simple approach: use the last available price
                if not hist.empty:
                    # Just use the first available price (simplified for now)
                    price = hist.iloc[0]['Close']
                    logger.info(f"Using price for {ticker}: {price}")
                    return float(price)
                
                logger.warning(f"No price data found for {ticker}")
                return None
                
                price_value = float(price)
                
                # Cache the result
                self.cache_manager.add_to_cache("price", cache_key, price_value)
                
                logger.info(f"Fetched price for {ticker}: {price_value}")
                return price_value
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch price for {ticker} after {MAX_API_RETRIES} attempts")
                    return None
        
        return None
    
    def fetch_stock_prices_batch(self, ticker_dates: List[Tuple[str, datetime]]) -> Dict[str, Optional[float]]:
        """
        Fetch stock prices for multiple tickers and dates in batch.
        
        Args:
            ticker_dates: List of (ticker, date) tuples
            
        Returns:
            Dictionary mapping ticker_date_key to price or None
        """
        results = {}
        batch_groups = {}
        
        # Group by date to minimize API calls
        for ticker, date in ticker_dates:
            date_key = date.strftime('%Y-%m')
            if date_key not in batch_groups:
                batch_groups[date_key] = []
            batch_groups[date_key].append((ticker, date))
        
        # Process each date group
        for date_key, group in batch_groups.items():
            logger.info(f"Processing batch for {date_key}: {len(group)} tickers")
            
            # Split into smaller batches
            for i in range(0, len(group), self.batch_size):
                batch = group[i:i + self.batch_size]
                batch_results = self._fetch_batch_prices(batch)
                results.update(batch_results)
                
                # Rate limiting
                if i + self.batch_size < len(group):
                    time.sleep(1)  # 1 second delay between batches
        
        return results
    
    def _fetch_batch_prices(self, ticker_dates: List[Tuple[str, datetime]]) -> Dict[str, Optional[float]]:
        """Fetch prices for a batch of tickers."""
        results = {}
        tickers_to_fetch = []
        
        # Check cache first
        for ticker, date in ticker_dates:
            cache_key = f"{ticker}_{date.strftime('%Y-%m-%d')}"
            cached_price = self.cache_manager.get_from_cache("price", cache_key)
            
            if cached_price is not None:
                results[cache_key] = float(cached_price)
            else:
                tickers_to_fetch.append((ticker, date, cache_key))
        
        if not tickers_to_fetch:
            return results
        
        # Group tickers by formatted ticker for batch download
        ticker_groups = {}
        for ticker, date, cache_key in tickers_to_fetch:
            formatted_ticker = self.format_ticker_for_yfinance(ticker)
            if formatted_ticker:
                if formatted_ticker not in ticker_groups:
                    ticker_groups[formatted_ticker] = []
                ticker_groups[formatted_ticker].append((ticker, date, cache_key))
        
        # Fetch each group
        for formatted_ticker, group in ticker_groups.items():
            try:
                # Get date range for this group
                dates = [date for _, date, _ in group]
                start_date = min(dates) - timedelta(days=30)
                end_date = max(dates) + timedelta(days=1)
                
                logger.info(f"Fetching batch for {formatted_ticker} from {start_date.date()} to {end_date.date()}")
                
                # Download historical data
                stock = yf.Ticker(formatted_ticker)
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    timeout=YFINANCE_TIMEOUT
                )
                
                if hist.empty:
                    logger.warning(f"No price data found for {formatted_ticker}")
                    for _, _, cache_key in group:
                        results[cache_key] = None
                    continue
                
                # Process each ticker in the group
                for ticker, date, cache_key in group:
                    # Find price for the specific date
                    target_date = date
                    
                    # Try to find exact date or closest date
                    hist_dates = hist.index
                    date_diffs = [(abs((d.date() - target_date.date()).days), d) for d in hist_dates]
                    date_diffs.sort(key=lambda x: x[0])
                    
                    if date_diffs and date_diffs[0][0] <= 30:  # Within 30 days
                        closest_date = date_diffs[0][1]
                        price = hist.loc[closest_date]['Close']
                        
                        if pd.notna(price):
                            price_value = float(price)
                            results[cache_key] = price_value
                            self.cache_manager.add_to_cache("price", cache_key, price_value)
                            logger.debug(f"Found price for {ticker} on {date.date()}: {price_value}")
                        else:
                            results[cache_key] = None
                            logger.warning(f"No valid price for {ticker} on {date.date()}")
                    else:
                        results[cache_key] = None
                        logger.warning(f"No price data near {date.date()} for {ticker}")
                        
            except Exception as e:
                logger.error(f"Error fetching batch for {formatted_ticker}: {e}")
                for _, _, cache_key in group:
                    results[cache_key] = None
        
        return results
    
    def _fetch_benchmark_price(self, benchmark: str, date: datetime) -> Optional[float]:
        """
        Fetch benchmark index price.
        
        Args:
            benchmark: Benchmark name (KOSPI, S&P, SPX)
            date: Date to fetch price for
            
        Returns:
            Benchmark price or None if not found
        """
        cache_key = f"{benchmark}_{date.strftime('%Y-%m-%d')}"
        
        # Try cache first
        cached_price = self.cache_manager.get_from_cache("price", cache_key)
        if cached_price is not None:
            return float(cached_price)
        
        # Map benchmark to yfinance ticker
        benchmark_tickers = {
            "KOSPI": "^KS11",  # KOSPI Composite Index
            "S&P": "^GSPC",    # S&P 500
            "SPX": "^GSPC"     # S&P 500 (alternative)
        }
        
        yf_ticker = benchmark_tickers.get(benchmark)
        if not yf_ticker:
            logger.warning(f"Unknown benchmark: {benchmark}")
            return None
        
        for attempt in range(MAX_API_RETRIES):
            try:
                logger.info(f"Fetching benchmark price for {benchmark} ({yf_ticker}) on {date}")
                
                # Download historical data
                index = yf.Ticker(yf_ticker)
                
                # Get data for the month containing the date
                start_date = date - timedelta(days=30)
                end_date = date + timedelta(days=1)
                
                hist = index.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    timeout=YFINANCE_TIMEOUT
                )
                
                if hist.empty:
                    logger.warning(f"No price data found for {benchmark} around {date}")
                    return None
                
                # Find the closest date to our target date
                hist = hist.sort_index()
                
                # Simple approach: use the last available price
                if not hist.empty:
                    # Just use the first available price (simplified for now)
                    price = hist.iloc[0]['Close']
                    logger.info(f"Using price for {benchmark}: {price}")
                    return float(price)
                
                logger.warning(f"No price data found for {benchmark}")
                return None
                
                price_value = float(price)
                
                # Cache the result
                self.cache_manager.add_to_cache("price", cache_key, price_value)
                
                logger.info(f"Fetched benchmark price for {benchmark}: {price_value}")
                return price_value
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {benchmark}: {e}")
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch benchmark price for {benchmark} after {MAX_API_RETRIES} attempts")
                    return None
        
        return None
    
    def fetch_stock_sector(self, ticker: str) -> Optional[str]:
        """
        Fetch sector information for a stock using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Sector name or None if not found
        """
        cache_key = ticker
        
        # Try cache first
        cached_sector = self.cache_manager.get_from_cache("sector", cache_key)
        if cached_sector is not None:
            return cached_sector
        
        formatted_ticker = self.format_ticker_for_yfinance(ticker)
        
        for attempt in range(MAX_API_RETRIES):
            try:
                logger.info(f"Fetching sector for {ticker} ({formatted_ticker})")
                
                stock = yf.Ticker(formatted_ticker)
                info = stock.info
                
                sector = info.get('sector')
                industry = info.get('industry')
                
                if sector:
                    # Cache the result
                    self.cache_manager.add_to_cache("sector", cache_key, sector)
                    
                    logger.info(f"Fetched sector for {ticker}: {sector} (Industry: {industry})")
                    return sector
                else:
                    logger.warning(f"No sector found for {ticker}")
                    return None
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker} sector: {e}")
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch sector for {ticker} after {MAX_API_RETRIES} attempts")
                    return None
        
        return None
    
    def fetch_sectors_batch(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """
        Fetch sectors for multiple tickers in batch.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to sector or None
        """
        results = {}
        tickers_to_fetch = []
        
        # Check cache first
        for ticker in tickers:
            cached_sector = self.cache_manager.get_from_cache("sector", ticker)
            if cached_sector is not None:
                results[ticker] = cached_sector
            else:
                tickers_to_fetch.append(ticker)
        
        if not tickers_to_fetch:
            return results
        
        # Process in batches
        for i in range(0, len(tickers_to_fetch), self.batch_size):
            batch = tickers_to_fetch[i:i + self.batch_size]
            batch_results = self._fetch_batch_sectors(batch)
            results.update(batch_results)
            
            # Rate limiting
            if i + self.batch_size < len(tickers_to_fetch):
                time.sleep(1)  # 1 second delay between batches
        
        return results
    
    def _fetch_batch_sectors(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """Fetch sectors for a batch of tickers."""
        results = {}
        
        for ticker in tickers:
            try:
                formatted_ticker = self.format_ticker_for_yfinance(ticker)
                if not formatted_ticker:
                    results[ticker] = None
                    continue
                
                logger.debug(f"Fetching sector for {ticker} ({formatted_ticker})")
                
                stock = yf.Ticker(formatted_ticker)
                info = stock.info
                
                sector = info.get('sector')
                if sector:
                    results[ticker] = sector
                    self.cache_manager.add_to_cache("sector", ticker, sector)
                    logger.debug(f"Found sector for {ticker}: {sector}")
                else:
                    results[ticker] = None
                    logger.warning(f"No sector found for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching sector for {ticker}: {e}")
                results[ticker] = None
        
        return results
    
    def fetch_exchange_rate(self, from_currency: str, to_currency: str, 
                           date: datetime) -> Optional[float]:
        """
        Fetch exchange rate for a specific date.
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Date to fetch rate for
            
        Returns:
            Exchange rate or None if not found
        """
        # Handle same currency
        if from_currency == to_currency:
            return 1.0
        
        cache_key = f"{from_currency}_{to_currency}_{date.strftime('%Y-%m-%d')}"
        
        # Try cache first
        cached_rate = self.cache_manager.get_from_cache("fx", cache_key)
        if cached_rate is not None:
            return float(cached_rate)
        
        # Try multiple API sources
        rate = self._fetch_from_exchangerate_api(from_currency, to_currency, date)
        
        if rate is None:
            rate = self._fetch_from_ecb(from_currency, to_currency, date)
        
        if rate is not None:
            # Cache the result
            self.cache_manager.add_to_cache("fx", cache_key, rate)
            logger.info(f"Fetched exchange rate {from_currency}/{to_currency} on {date}: {rate}")
        else:
            logger.warning(f"Failed to fetch exchange rate {from_currency}/{to_currency} on {date}")
        
        return rate
    
    def _fetch_from_exchangerate_api(self, from_currency: str, to_currency: str,
                                    date: datetime) -> Optional[float]:
        """Fetch exchange rate from exchangerate-api.com."""
        try:
            # exchangerate-api.com provides latest rates, not historical
            # For historical data, we'd need a different source
            url = EXCHANGERATE_API_URL.format(base_currency=from_currency)
            
            response = self.session.get(url, timeout=EXCHANGERATE_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            rates = data.get('rates', {})
            
            if to_currency in rates:
                return float(rates[to_currency])
            else:
                logger.warning(f"Currency {to_currency} not found in exchange rate API response")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to fetch from exchangerate-api.com: {e}")
            return None
    
    def _fetch_from_ecb(self, from_currency: str, to_currency: str,
                       date: datetime) -> Optional[float]:
        """Fetch exchange rate from European Central Bank (historical data)."""
        try:
            # ECB provides EUR-based rates
            if from_currency == "EUR":
                # Get EUR to target currency
                url = f"https://api.exchangerate.host/{date.strftime('%Y-%m-%d')}"
                params = {
                    "base": "EUR",
                    "symbols": to_currency
                }
                
                response = self.session.get(url, params=params, timeout=EXCHANGERATE_TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                if data.get('success', False):
                    rates = data.get('rates', {})
                    if to_currency in rates:
                        return float(rates[to_currency])
            
            # For non-EUR currencies, we need to convert through EUR
            # This is simplified - in production you'd want a more robust solution
            logger.debug(f"ECB API doesn't support {from_currency} directly")
            return None
            
        except Exception as e:
            logger.debug(f"ECB API failed: {e}")
            return None
    
    def fetch_batch_prices(self, ticker_dates: List[Tuple[str, datetime]]) -> Dict[str, float]:
        """
        Fetch prices for multiple tickers in batch.
        
        Args:
            ticker_dates: List of (ticker, date) tuples
            
        Returns:
            Dictionary mapping ticker_date to price
        """
        results = {}
        
        for ticker, date in ticker_dates:
            price = self.fetch_stock_price(ticker, date)
            if price is not None:
                key = f"{ticker}_{date.strftime('%Y-%m-%d')}"
                results[key] = price
        
        return results
    
    def fetch_batch_sectors(self, tickers: List[str]) -> Dict[str, str]:
        """
        Fetch sectors for multiple tickers in batch.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to sector
        """
        results = {}
        
        for ticker in tickers:
            sector = self.fetch_stock_sector(ticker)
            if sector is not None:
                results[ticker] = sector
        
        return results
    
    def estimate_missing_price(self, ticker: str, date: datetime, 
                              portfolio_data: pd.DataFrame) -> Optional[float]:
        """
        Estimate missing price using available data.
        
        Args:
            ticker: Stock ticker
            date: Date for estimation
            portfolio_data: Full portfolio data for context
            
        Returns:
            Estimated price or None
        """
        try:
            # Get historical prices for this ticker
            ticker_data = portfolio_data[portfolio_data['ticker'] == ticker].copy()
            if 'date' in ticker_data.columns:
                ticker_data = ticker_data.sort_values(by=['date'])
            
            if ticker_data.empty:
                logger.warning(f"No historical data for {ticker} to estimate price")
                return None
            
            # Find closest available price before the target date
            past_prices = ticker_data[ticker_data['date'] < date]
            
            if not past_prices.empty:
                # Use most recent past price
                latest_past = past_prices.iloc[-1]
                return float(latest_past['price'])
            
            # If no past prices, use earliest future price
            future_prices = ticker_data[ticker_data['date'] > date]
            if not future_prices.empty:
                earliest_future = future_prices.iloc[0]
                return float(earliest_future['price'])
            
            logger.warning(f"No data available to estimate price for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error estimating price for {ticker}: {e}")
            return None
    
    def estimate_missing_exchange_rate(self, currency: str, date: datetime,
                                      portfolio_data: pd.DataFrame) -> Optional[float]:
        """
        Estimate missing exchange rate using available data.
        
        Args:
            currency: Currency code
            date: Date for estimation
            portfolio_data: Full portfolio data for context
            
        Returns:
            Estimated exchange rate or None
        """
        try:
            # Get historical exchange rates for this currency
            currency_data = portfolio_data[portfolio_data['currency'] == currency].copy()
            currency_data = currency_data.sort_values(by='date', ascending=True)
            
            if currency_data.empty:
                logger.warning(f"No historical data for {currency} to estimate exchange rate")
                return None
            
            # Find closest available rate before the target date
            past_rates = currency_data[currency_data['date'] < date]
            
            if not past_rates.empty:
                # Use most recent past rate
                latest_past = past_rates.iloc[-1]
                return float(latest_past['exchange_rate'])
            
            # If no past rates, use earliest future rate
            future_rates = currency_data[currency_data['date'] > date]
            if not future_rates.empty:
                earliest_future = future_rates.iloc[0]
                return float(earliest_future['exchange_rate'])
            
            logger.warning(f"No data available to estimate exchange rate for {currency}")
            return None
            
        except Exception as e:
            logger.error(f"Error estimating exchange rate for {currency}: {e}")
            return None
    
    def optimize_performance(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze portfolio data and provide performance optimization recommendations.
        
        Args:
            portfolio_data: Portfolio data to analyze
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "api_calls_needed": 0,
            "cache_hits_possible": 0,
            "batch_optimizations": [],
            "memory_usage": {},
            "processing_time_estimate": 0
        }
        
        try:
            # Analyze data for optimization opportunities
            if portfolio_data is not None and not portfolio_data.empty:
                # Count unique tickers and dates
                unique_tickers = portfolio_data['ticker'].nunique() if 'ticker' in portfolio_data.columns else 0
                unique_dates = portfolio_data['date'].nunique() if 'date' in portfolio_data.columns else 0
                unique_currencies = portfolio_data['currency'].nunique() if 'currency' in portfolio_data.columns else 0
                
                # Estimate API calls needed
                recommendations["api_calls_needed"] = unique_tickers * unique_dates
                
                # Estimate cache hits
                recommendations["cache_hits_possible"] = int(recommendations["api_calls_needed"] * 0.3)  # Assume 30% cache hit rate
                
                # Batch optimization recommendations
                if unique_tickers > 5:
                    recommendations["batch_optimizations"].append(
                        f"Use batch processing for {unique_tickers} tickers (group by date)"
                    )
                
                if unique_dates > 3:
                    recommendations["batch_optimizations"].append(
                        f"Group API calls by {unique_dates} dates to minimize requests"
                    )
                
                # Memory usage estimation
                estimated_memory_mb = (len(portfolio_data) * 100) / (1024 * 1024)  # Rough estimate
                recommendations["memory_usage"] = {
                    "estimated_mb": round(estimated_memory_mb, 2),
                    "rows": len(portfolio_data),
                    "columns": len(portfolio_data.columns)
                }
                
                # Processing time estimate (seconds)
                base_time = unique_tickers * 0.5  # 0.5 seconds per ticker
                recommendations["processing_time_estimate"] = round(base_time, 1)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return recommendations
    
    def save_caches(self):
        """Save all cache data to files."""
        self.cache_manager.save_all_caches()
        logger.info("Saved all cache files")
    
    def clear_expired_cache(self):
        """Clear expired cache entries."""
        self.cache_manager.clear_expired_entries()


def test_data_fetcher():
    """Test function for the DataFetcher class."""
    import sys
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fetcher = DataFetcher()
    
    print("Testing DataFetcher...")
    
    # Test 1: Fetch stock price
    print("\n1. Testing stock price fetch for BABA:")
    test_date = datetime(2025, 12, 31)
    price = fetcher.fetch_stock_price("BABA", test_date)
    print(f"   Price for BABA on {test_date}: {price}")
    
    # Test 2: Fetch sector
    print("\n2. Testing sector fetch for BABA:")
    sector = fetcher.fetch_stock_sector("BABA")
    print(f"   Sector for BABA: {sector}")
    
    # Test 3: Fetch exchange rate
    print("\n3. Testing exchange rate fetch (USD to KRW):")
    rate = fetcher.fetch_exchange_rate("USD", "KRW", test_date)
    print(f"   USD/KRW rate on {test_date}: {rate}")
    
    # Test 4: Test ticker formatting
    print("\n4. Testing ticker formatting:")
    test_tickers = ["035900.KQ", "0700.HK", "2801 JT", "RMS.PA", "BABA"]
    for ticker in test_tickers:
        formatted = fetcher.format_ticker_for_yfinance(ticker)
        print(f"   {ticker} -> {formatted}")
    
    # Save caches
    fetcher.save_caches()
    
    print("\nDataFetcher test completed!")


if __name__ == "__main__":
    test_data_fetcher()