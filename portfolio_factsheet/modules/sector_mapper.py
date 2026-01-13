"""
Sector mapper module for detecting and mapping missing sector data.
Uses yfinance for sector information and maps to standard GICS sectors.
"""

import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from config import SECTOR_MAPPING, SECTOR_CACHE_FILE, VALID_SECTORS
from .data_fetcher import DataFetcher

# Set up logging
logger = logging.getLogger(__name__)


class SectorMapper:
    """Maps and manages sector information for portfolio holdings."""
    
    def __init__(self, data_fetcher: Optional[DataFetcher] = None):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.sector_cache = self._load_sector_cache()
        self.reverse_sector_mapping = {v: k for k, v in SECTOR_MAPPING.items()}
        
    def _load_sector_cache(self) -> Dict:
        """Load sector cache from file."""
        try:
            if os.path.exists(SECTOR_CACHE_FILE):
                with open(SECTOR_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # Convert from CacheManager format to simple dict
                    simple_cache = {}
                    for ticker, entry in cache_data.items():
                        if isinstance(entry, dict) and 'value' in entry:
                            simple_cache[ticker] = {
                                'sector': entry['value'],
                                'last_updated': entry.get('timestamp', ''),
                                'source': 'cache'
                            }
                        else:
                            # Old format or already simple
                            simple_cache[ticker] = entry
                    return simple_cache
            return {}
        except Exception as e:
            logger.error(f"Error loading sector cache: {e}")
            return {}
    
    def _save_sector_cache(self):
        """Save sector cache to file."""
        try:
            with open(SECTOR_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.sector_cache, f, indent=2, ensure_ascii=False)
            logger.debug("Saved sector cache")
        except Exception as e:
            logger.error(f"Error saving sector cache: {e}")
    
    def get_sector_for_ticker(self, ticker: str, date: datetime = None) -> Optional[str]:
        """
        Get sector for a ticker, using cache first then yfinance.
        
        Args:
            ticker: Stock ticker symbol
            date: Date for sector information (not used for caching, but for logging)
            
        Returns:
            Sector name or None if not found
        """
        # Check cache first
        if ticker in self.sector_cache:
            cached_entry = self.sector_cache[ticker]
            logger.debug(f"Cache hit for {ticker}: {cached_entry.get('sector')}")
            return cached_entry.get('sector')
        
        # Fetch from yfinance
        if self.data_fetcher:
            sector = self.data_fetcher.fetch_stock_sector(ticker)
            if sector:
                # Map to standard sector if needed
                mapped_sector = self.map_to_standard_sector(sector)
                
                # Cache the result
                self.sector_cache[ticker] = {
                    "sector": mapped_sector,
                    "original_sector": sector,
                    "last_updated": datetime.now().isoformat(),
                    "source": "yfinance"
                }
                self._save_sector_cache()
                
                logger.info(f"Mapped sector for {ticker}: {sector} -> {mapped_sector}")
                return mapped_sector
        
        logger.warning(f"No sector found for {ticker}")
        return None
    
    def map_to_standard_sector(self, sector: str) -> str:
        """
        Map a sector name to standard GICS sector.
        
        Args:
            sector: Original sector name from yfinance or other source
            
        Returns:
            Standardized sector name
        """
        if not sector:
            return "Unknown"
        
        # Direct mapping
        if sector in SECTOR_MAPPING:
            return SECTOR_MAPPING[sector]
        
        # Case-insensitive matching
        sector_lower = sector.lower()
        for yf_sector, std_sector in SECTOR_MAPPING.items():
            if yf_sector.lower() in sector_lower or sector_lower in yf_sector.lower():
                return std_sector
        
        # Check if already a standard sector
        if sector in VALID_SECTORS:
            return sector
        
        # Try to match by keywords
        sector_keywords = {
            "tech": "Information Technology",
            "technology": "Information Technology",
            "software": "Information Technology",
            "internet": "Information Technology",
            "communication": "Communication Services",
            "telecom": "Communication Services",
            "media": "Communication Services",
            "consumer": "Consumer Discretionary",  # Default to discretionary
            "retail": "Consumer Discretionary",
            "automotive": "Consumer Discretionary",
            "food": "Consumer Staples",
            "beverage": "Consumer Staples",
            "health": "Health Care",
            "medical": "Health Care",
            "pharma": "Health Care",
            "financial": "Financials",
            "bank": "Financials",
            "insurance": "Financials",
            "industrial": "Industrials",
            "manufacturing": "Industrials",
            "transport": "Industrials",
            "energy": "Energy",
            "oil": "Energy",
            "gas": "Energy",
            "utility": "Utilities",
            "real estate": "Real Estate",
            "materials": "Materials",
            "mining": "Materials",
            "fixed income": "Fixed Income",
            "bond": "Fixed Income"
        }
        
        for keyword, mapped_sector in sector_keywords.items():
            if keyword in sector_lower:
                return mapped_sector
        
        logger.warning(f"Could not map sector: {sector}")
        return sector  # Return original if no mapping found
    
    def get_sectors_for_tickers(self, tickers: List[str]) -> Dict[str, str]:
        """
        Get sectors for multiple tickers in batch.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to sector
        """
        results = {}
        
        for ticker in tickers:
            sector = self.get_sector_for_ticker(ticker)
            if sector:
                results[ticker] = sector
        
        return results
    
    def infer_sector_from_history(self, ticker: str, portfolio_data: pd.DataFrame) -> Optional[str]:
        """
        Infer sector from historical data in the portfolio.
        
        Args:
            ticker: Stock ticker symbol
            portfolio_data: Full portfolio data for context
            
        Returns:
            Inferred sector or None
        """
        try:
            # Look for this ticker in historical data
            ticker_history = portfolio_data[portfolio_data['ticker'] == ticker]
            
            if not ticker_history.empty:
                # Check if sector is available in any historical record
                available_sectors = ticker_history['sector'].dropna().unique()
                if len(available_sectors) > 0:
                    # Use the most common sector
                    sector_counts = ticker_history['sector'].value_counts()
                    most_common = sector_counts.index[0]
                    logger.info(f"Inferred sector for {ticker} from history: {most_common}")
                    return most_common
            
            # Try to infer from similar tickers (same company, different suffix)
            base_ticker = self._get_base_ticker(ticker)
            if base_ticker != ticker:
                similar_tickers = portfolio_data[
                    portfolio_data['ticker'].str.contains(base_ticker)
                ]
                if not similar_tickers.empty:
                    available_sectors = similar_tickers['sector'].dropna().unique()
                    if len(available_sectors) > 0:
                        sector_counts = similar_tickers['sector'].value_counts()
                        most_common = sector_counts.index[0]
                        logger.info(f"Inferred sector for {ticker} from similar tickers: {most_common}")
                        return most_common
            
            return None
            
        except Exception as e:
            logger.error(f"Error inferring sector for {ticker}: {e}")
            return None
    
    def _get_base_ticker(self, ticker: str) -> str:
        """Extract base ticker without exchange suffix."""
        # Remove common exchange suffixes
        suffixes = ['.KQ', '.KS', '.HK', '.SZ', '.JT', '.T', '.PA', '.SW', '.DE', '.L']
        for suffix in suffixes:
            if ticker.endswith(suffix):
                return ticker.replace(suffix, '')
        
        # Remove spaces and special cases
        if ' ' in ticker:
            return ticker.split(' ')[0]
        
        return ticker
    
    def validate_sector(self, sector: str) -> bool:
        """Validate if a sector is in the standard list."""
        return sector in VALID_SECTORS
    
    def get_sector_statistics(self, portfolio_data: pd.DataFrame) -> Dict:
        """
        Calculate sector statistics for the portfolio.
        
        Args:
            portfolio_data: Portfolio data with sector information
            
        Returns:
            Dictionary with sector statistics
        """
        try:
            # Get latest month's data
            latest_month = portfolio_data['year_month'].max()
            latest_data = portfolio_data[portfolio_data['year_month'] == latest_month]
            
            # Calculate sector weights
            sector_weights = {}
            total_value = 0
            
            for _, row in latest_data.iterrows():
                if pd.notna(row['sector']) and pd.notna(row['weight_pct']):
                    sector = row['sector']
                    weight = row['weight_pct']
                    
                    if sector not in sector_weights:
                        sector_weights[sector] = 0
                    sector_weights[sector] += weight
                    total_value += weight
            
            # Normalize weights to percentage
            if total_value > 0:
                sector_weights = {k: (v / total_value * 100) for k, v in sector_weights.items()}
            
            # Count holdings per sector
            sector_counts = latest_data['sector'].value_counts().to_dict()
            
            # Identify missing sectors
            missing_sectors = latest_data[latest_data['sector'].isna() | (latest_data['sector'] == '')]
            missing_tickers = missing_sectors['ticker'].tolist()
            
            return {
                "sector_weights": sector_weights,
                "sector_counts": sector_counts,
                "missing_sectors": missing_tickers,
                "total_sectors": len(sector_weights),
                "latest_month": str(latest_month)
            }
            
        except Exception as e:
            logger.error(f"Error calculating sector statistics: {e}")
            return {}
    
    def update_portfolio_sectors(self, portfolio_data: pd.DataFrame, 
                                missing_sectors_info: List[Dict]) -> pd.DataFrame:
        """
        Update portfolio data with missing sector information.
        
        Args:
            portfolio_data: Portfolio data to update
            missing_sectors_info: List of dictionaries with sector updates
            
        Returns:
            Updated portfolio data
        """
        df = portfolio_data.copy()
        
        for sector_info in missing_sectors_info:
            ticker = sector_info.get('ticker')
            sector = sector_info.get('sector')
            date = sector_info.get('date')
            
            if ticker and sector and date:
                mask = (df['ticker'] == ticker) & (df['date'] == date)
                df.loc[mask, 'sector'] = sector
                
                # Also update cache
                self.sector_cache[ticker] = {
                    "sector": sector,
                    "last_updated": datetime.now().isoformat(),
                    "source": "manual_update"
                }
        
        # Save updated cache
        self._save_sector_cache()
        
        logger.info(f"Updated {len(missing_sectors_info)} sector entries")
        return df


def test_sector_mapper():
    """Test function for the SectorMapper class."""
    import sys
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    mapper = SectorMapper()
    
    print("Testing SectorMapper...")
    
    # Test 1: Sector mapping
    print("\n1. Testing sector mapping:")
    test_sectors = [
        "Consumer Cyclical",
        "Consumer Defensive", 
        "Technology",
        "Healthcare",
        "Some Unknown Sector"
    ]
    
    for sector in test_sectors:
        mapped = mapper.map_to_standard_sector(sector)
        print(f"   {sector} -> {mapped}")
    
    # Test 2: Get sector for ticker
    print("\n2. Testing sector fetch for BABA:")
    sector = mapper.get_sector_for_ticker("BABA")
    print(f"   Sector for BABA: {sector}")
    
    # Test 3: Validate sectors
    print("\n3. Testing sector validation:")
    test_validation = ["Information Technology", "Consumer Discretionary", "Unknown Sector"]
    for sector in test_validation:
        valid = mapper.validate_sector(sector)
        print(f"   {sector}: {'Valid' if valid else 'Invalid'}")
    
    # Test 4: Base ticker extraction
    print("\n4. Testing base ticker extraction:")
    test_tickers = ["035900.KQ", "0700.HK", "2801 JT", "BABA"]
    for ticker in test_tickers:
        base = mapper._get_base_ticker(ticker)
        print(f"   {ticker} -> {base}")
    
    print("\nSectorMapper test completed!")


if __name__ == "__main__":
    test_sector_mapper()