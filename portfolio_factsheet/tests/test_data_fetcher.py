"""
Unit tests for data_fetcher module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import tempfile

from modules.data_fetcher import DataFetcher, CacheManager


class TestCacheManager(unittest.TestCase):
    """Test cases for CacheManager class."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_files = {
            "sector": os.path.join(self.temp_dir, "sector_cache.json"),
            "fx": os.path.join(self.temp_dir, "fx_cache.json"),
            "price": os.path.join(self.temp_dir, "price_cache.json")
        }
        
        # Patch config constants
        self.config_patcher = patch.dict('modules.data_fetcher.__dict__', {
            'SECTOR_CACHE_FILE': self.cache_files["sector"],
            'FX_CACHE_FILE': self.cache_files["fx"],
            'PRICE_CACHE_FILE': self.cache_files["price"],
            'CACHE_ENABLED': True,
            'CACHE_EXPIRY_DAYS': 7
        })
        self.config_patcher.start()
        
        self.cache_manager = CacheManager()
    
    def tearDown(self):
        """Clean up test files."""
        self.config_patcher.stop()
        for cache_file in self.cache_files.values():
            if os.path.exists(cache_file):
                os.unlink(cache_file)
        os.rmdir(self.temp_dir)
    
    def test_cache_creation(self):
        """Test cache creation and loading."""
        # Cache should be created if file doesn't exist
        self.assertIn("sector", self.cache_manager.cache_data)
        self.assertIn("fx", self.cache_manager.cache_data)
        self.assertIn("price", self.cache_manager.cache_data)
        
        # All caches should be empty dictionaries
        self.assertEqual(self.cache_manager.cache_data["sector"], {})
        self.assertEqual(self.cache_manager.cache_data["fx"], {})
        self.assertEqual(self.cache_manager.cache_data["price"], {})
    
    def test_cache_save_and_load(self):
        """Test saving and loading cache."""
        # Add test data to cache
        test_data = {
            "AAPL": {"value": "Technology", "timestamp": "2025-01-01T00:00:00"}
        }
        self.cache_manager.cache_data["sector"] = test_data
        
        # Save cache
        self.cache_manager.save_cache("sector")
        
        # Verify file exists
        self.assertTrue(os.path.exists(self.cache_files["sector"]))
        
        # Create new cache manager to test loading
        new_cache_manager = CacheManager()
        
        # Verify data was loaded
        self.assertEqual(new_cache_manager.cache_data["sector"], test_data)
    
    def test_get_from_cache(self):
        """Test retrieving data from cache."""
        # Add test entry with current timestamp
        current_time = datetime.now().isoformat()
        test_entry = {
            "value": "Technology",
            "timestamp": current_time
        }
        self.cache_manager.cache_data["sector"]["AAPL"] = test_entry
        
        # Get from cache
        result = self.cache_manager.get_from_cache("sector", "AAPL")
        self.assertEqual(result, "Technology")
    
    def test_get_from_cache_expired(self):
        """Test retrieving expired cache entry."""
        # Add test entry with old timestamp
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        test_entry = {
            "value": "Technology",
            "timestamp": old_time
        }
        self.cache_manager.cache_data["sector"]["AAPL"] = test_entry
        
        # Get from cache (should return None due to expiry)
        result = self.cache_manager.get_from_cache("sector", "AAPL")
        self.assertIsNone(result)
        
        # Entry should be removed from cache
        self.assertNotIn("AAPL", self.cache_manager.cache_data["sector"])
    
    def test_save_to_cache(self):
        """Test saving data to cache."""
        # Save to cache
        self.cache_manager.save_to_cache("sector", "AAPL", "Technology")
        
        # Verify data was saved
        self.assertIn("AAPL", self.cache_manager.cache_data["sector"])
        cache_entry = self.cache_manager.cache_data["sector"]["AAPL"]
        self.assertEqual(cache_entry["value"], "Technology")
        self.assertIn("timestamp", cache_entry)


class TestDataFetcher(unittest.TestCase):
    """Test cases for DataFetcher class."""
    
    def setUp(self):
        """Set up test data."""
        # Create temp directory for cache files
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch config constants
        self.config_patcher = patch.dict('modules.data_fetcher.__dict__', {
            'SECTOR_CACHE_FILE': os.path.join(self.temp_dir, "sector_cache.json"),
            'FX_CACHE_FILE': os.path.join(self.temp_dir, "fx_cache.json"),
            'PRICE_CACHE_FILE': os.path.join(self.temp_dir, "price_cache.json"),
            'CACHE_ENABLED': True,
            'CACHE_EXPIRY_DAYS': 7,
            'YFINANCE_TIMEOUT': 10,
            'EXCHANGERATE_API_URL': 'https://api.exchangerate-api.com/v4/latest/',
            'EXCHANGERATE_TIMEOUT': 10,
            'MAX_API_RETRIES': 3,
            'API_RETRY_DELAY': 1,
            'TICKER_MAPPING': {
                '005930': '005930.KS',
                '000660': '000660.KS',
                'KOSPI': '^KS11'
            }
        })
        self.config_patcher.start()
        
        self.data_fetcher = DataFetcher()
    
    def tearDown(self):
        """Clean up test files."""
        self.config_patcher.stop()
        for file in os.listdir(self.temp_dir):
            os.unlink(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    @patch('modules.data_fetcher.yf.download')
    def test_fetch_stock_price_success(self, mock_download):
        """Test successful stock price fetch."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Close': [70000.0]
        }, index=[pd.Timestamp('2025-01-01')])
        mock_download.return_value = mock_data
        
        price = self.data_fetcher.fetch_stock_price('005930', '2025-01-01')
        
        self.assertEqual(price, 70000.0)
        mock_download.assert_called_once_with(
            '005930.KS',
            start='2024-12-31',
            end='2025-01-02',
            progress=False
        )
    
    @patch('modules.data_fetcher.yf.download')
    def test_fetch_stock_price_no_data(self, mock_download):
        """Test stock price fetch with no data."""
        # Mock empty response
        mock_download.return_value = pd.DataFrame()
        
        price = self.data_fetcher.fetch_stock_price('005930', '2025-01-01')
        
        self.assertIsNone(price)
    
    @patch('modules.data_fetcher.yf.download')
    def test_fetch_stock_price_exception(self, mock_download):
        """Test stock price fetch with exception."""
        # Mock exception
        mock_download.side_effect = Exception("API Error")
        
        price = self.data_fetcher.fetch_stock_price('005930', '2025-01-01')
        
        self.assertIsNone(price)
    
    @patch('modules.data_fetcher.requests.get')
    def test_fetch_exchange_rate_success(self, mock_get):
        """Test successful exchange rate fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'rates': {
                'KRW': 1300.0
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        rate = self.data_fetcher.fetch_exchange_rate('USD', 'KRW', '2025-01-01')
        
        self.assertEqual(rate, 1300.0)
    
    @patch('modules.data_fetcher.requests.get')
    def test_fetch_exchange_rate_no_data(self, mock_get):
        """Test exchange rate fetch with no data."""
        # Mock API response without target currency
        mock_response = Mock()
        mock_response.json.return_value = {
            'rates': {
                'JPY': 150.0
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        rate = self.data_fetcher.fetch_exchange_rate('USD', 'KRW', '2025-01-01')
        
        self.assertIsNone(rate)
    
    @patch('modules.data_fetcher.requests.get')
    def test_fetch_exchange_rate_exception(self, mock_get):
        """Test exchange rate fetch with exception."""
        # Mock exception
        mock_get.side_effect = Exception("API Error")
        
        rate = self.data_fetcher.fetch_exchange_rate('USD', 'KRW', '2025-01-01')
        
        self.assertIsNone(rate)
    
    def test_ticker_mapping(self):
        """Test ticker mapping functionality."""
        # Test Korean stock ticker
        mapped_ticker = self.data_fetcher._map_ticker('005930')
        self.assertEqual(mapped_ticker, '005930.KS')
        
        # Test benchmark ticker
        mapped_ticker = self.data_fetcher._map_ticker('KOSPI')
        self.assertEqual(mapped_ticker, '^KS11')
        
        # Test unmapped ticker (should return as-is)
        mapped_ticker = self.data_fetcher._map_ticker('AAPL')
        self.assertEqual(mapped_ticker, 'AAPL')
    
    def test_cache_usage(self):
        """Test that cache is used for repeated requests."""
        # First call should not be in cache
        with patch.object(self.data_fetcher.cache_manager, 'get_from_cache') as mock_get_cache:
            mock_get_cache.return_value = None
            with patch('modules.data_fetcher.yf.download') as mock_download:
                mock_data = pd.DataFrame({
                    'Close': [70000.0]
                }, index=[pd.Timestamp('2025-01-01')])
                mock_download.return_value = mock_data
                
                price = self.data_fetcher.fetch_stock_price('005930', '2025-01-01')
                
                # Cache should be checked
                mock_get_cache.assert_called_once_with('price', '005930_2025-01-01')
        
        # Second call should use cache
        with patch.object(self.data_fetcher.cache_manager, 'get_from_cache') as mock_get_cache:
            mock_get_cache.return_value = 70000.0
            with patch('modules.data_fetcher.yf.download') as mock_download:
                price = self.data_fetcher.fetch_stock_price('005930', '2025-01-01')
                
                # Cache should be used, no API call
                mock_download.assert_not_called()
                self.assertEqual(price, 70000.0)


if __name__ == '__main__':
    unittest.main()