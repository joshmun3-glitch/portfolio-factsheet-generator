"""
Unit tests for data_loader module.
"""

import unittest
import pandas as pd
import tempfile
import os
from datetime import datetime

from modules.data_loader import PortfolioData


class TestPortfolioData(unittest.TestCase):
    """Test cases for PortfolioData class."""
    
    def setUp(self):
        """Set up test data."""
        self.portfolio = PortfolioData()
        
        # Create sample CSV data
        self.sample_data = pd.DataFrame({
            '기준일': ['2025-01-01', '2025-01-01', '2025-02-01', '2025-02-01'],
            '종목코드': ['005930', '000660', '005930', '000660'],
            '종목명': ['삼성전자', 'SK하이닉스', '삼성전자', 'SK하이닉스'],
            '수량': [100, 50, 100, 50],
            '현재가': [70000, 150000, 72000, 155000],
            '통화': ['KRW', 'KRW', 'KRW', 'KRW'],
            '섹터': ['Technology', 'Technology', 'Technology', 'Technology']
        })
    
    def test_load_csv_success(self):
        """Test successful CSV loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False, encoding='utf-8-sig')
            temp_file = f.name
        
        try:
            success = self.portfolio.load_csv(temp_file)
            self.assertTrue(success)
            self.assertIsNotNone(self.portfolio.raw_data)
            self.assertIsNotNone(self.portfolio.processed_data)
            self.assertEqual(len(self.portfolio.processed_data), 4)
        finally:
            os.unlink(temp_file)
    
    def test_load_csv_missing_columns(self):
        """Test CSV loading with missing required columns."""
        invalid_data = pd.DataFrame({
            'date': ['2025-01-01'],
            'ticker': ['005930'],
            'price': [70000]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False, encoding='utf-8-sig')
            temp_file = f.name
        
        try:
            success = self.portfolio.load_csv(temp_file)
            self.assertFalse(success)
            self.assertGreater(len(self.portfolio.data_quality_issues), 0)
        finally:
            os.unlink(temp_file)
    
    def test_data_processing(self):
        """Test data processing functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False, encoding='utf-8-sig')
            temp_file = f.name
        
        try:
            success = self.portfolio.load_csv(temp_file)
            self.assertTrue(success)
            
            # Check column names are converted
            expected_columns = ['date', 'ticker', 'name', 'quantity', 'price', 
                              'currency', 'sector', 'value_local', 'value_krw']
            
            for col in expected_columns:
                self.assertIn(col, self.portfolio.processed_data.columns)
            
            # Check value calculations
            df = self.portfolio.processed_data
            self.assertIn('value_local', df.columns)
            self.assertIn('value_krw', df.columns)
            
            # Check date extraction
            self.assertGreater(len(self.portfolio.monthly_dates), 0)
            
        finally:
            os.unlink(temp_file)
    
    def test_missing_data_identification(self):
        """Test missing data identification."""
        # Create data with missing values
        missing_data = pd.DataFrame({
            '기준일': ['2025-01-01', '2025-01-01'],
            '종목코드': ['005930', '000660'],
            '종목명': ['삼성전자', 'SK하이닉스'],
            '수량': [100, 50],
            '현재가': [70000, None],  # Missing price
            '통화': ['KRW', 'USD'],
            '섹터': ['Technology', '']  # Missing sector
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            missing_data.to_csv(f.name, index=False, encoding='utf-8-sig')
            temp_file = f.name
        
        try:
            success = self.portfolio.load_csv(temp_file)
            self.assertTrue(success)
            
            # Check missing data identification
            self.assertGreater(len(self.portfolio.missing_data["prices"]), 0)
            self.assertGreater(len(self.portfolio.missing_data["sectors"]), 0)
            
        finally:
            os.unlink(temp_file)
    
    def test_data_quality_validation(self):
        """Test data quality validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False, encoding='utf-8-sig')
            temp_file = f.name
        
        try:
            success = self.portfolio.load_csv(temp_file)
            self.assertTrue(success)
            
            # Check data quality issues (should be minimal for valid data)
            self.assertIsInstance(self.portfolio.data_quality_issues, list)
            
            # Get quality report
            report = self.portfolio.get_data_quality_report()
            self.assertIn('summary', report)
            self.assertIn('issues', report)
            self.assertIn('recommendations', report)
            self.assertIn('missing_data', report)
            
        finally:
            os.unlink(temp_file)
    
    def test_portfolio_summary(self):
        """Test portfolio summary calculation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False, encoding='utf-8-sig')
            temp_file = f.name
        
        try:
            success = self.portfolio.load_csv(temp_file)
            self.assertTrue(success)
            
            summary = self.portfolio.portfolio_summary
            self.assertIn('total_rows', summary)
            self.assertIn('unique_tickers', summary)
            self.assertIn('unique_dates', summary)
            self.assertIn('total_value_krw', summary)
            
            # Check specific values
            self.assertEqual(summary['total_rows'], 4)
            self.assertEqual(summary['unique_tickers'], 2)
            self.assertEqual(summary['unique_dates'], 2)
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()