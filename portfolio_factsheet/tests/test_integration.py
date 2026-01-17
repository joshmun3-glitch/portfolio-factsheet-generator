"""
Integration tests for the complete portfolio factsheet application workflow.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

# Add modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import PortfolioData
from modules.data_fetcher import DataFetcher
from modules.sector_mapper import SectorMapper
from modules.portfolio_calc import PortfolioCalculator
from modules.report_gen import ReportGenerator


class TestCompleteWorkflow(unittest.TestCase):
    """Test the complete portfolio factsheet generation workflow."""
    
    def setUp(self):
        """Set up test data for complete workflow."""
        # Create temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample CSV data in Korean format
        self.sample_csv_data = pd.DataFrame({
            '기준일': ['2025-01-01', '2025-01-01', '2025-02-01', '2025-02-01'],
            '종목코드': ['005930', '000660', '005930', '000660'],
            '종목명': ['삼성전자', 'SK하이닉스', '삼성전자', 'SK하이닉스'],
            '수량': [100, 50, 100, 50],
            '현재가': [70000, 150000, 72000, 155000],
            '통화': ['KRW', 'KRW', 'KRW', 'KRW'],
            '섹터': ['Technology', 'Technology', 'Technology', 'Technology']
        })
        
        # Save to temp CSV file
        self.csv_file = os.path.join(self.temp_dir, 'test_portfolio.csv')
        self.sample_csv_data.to_csv(self.csv_file, index=False, encoding='utf-8-sig')
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow_with_mock_apis(self):
        """Test complete workflow using mocked API calls."""
        # Step 1: Load portfolio data
        portfolio = PortfolioData()
        success = portfolio.load_csv(self.csv_file)
        
        self.assertTrue(success, "Failed to load CSV file")
        self.assertIsNotNone(portfolio.raw_data)
        self.assertIsNotNone(portfolio.processed_data)
        
        # Check data processing
        self.assertEqual(len(portfolio.processed_data), 4)
        self.assertIn('value_krw', portfolio.processed_data.columns)
        
        # Step 2: Mock data fetching
        with patch('modules.data_fetcher.yf.download') as mock_yf, \
             patch('modules.data_fetcher.requests.get') as mock_requests:
            
            # Mock yfinance response for stock prices
            mock_yf.return_value = pd.DataFrame({
                'Close': [70000.0, 72000.0]
            }, index=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-02-01')])
            
            # Mock exchange rate API response
            mock_response = Mock()
            mock_response.json.return_value = {'rates': {'KRW': 1.0}}
            mock_response.raise_for_status.return_value = None
            mock_requests.return_value = mock_response
            
            # Initialize data fetcher
            data_fetcher = DataFetcher()
            
            # Test fetching stock price
            price = data_fetcher.fetch_stock_price('005930', '2025-01-01')
            self.assertEqual(price, 70000.0)
            
            # Test fetching exchange rate
            rate = data_fetcher.fetch_exchange_rate('KRW', 'KRW', '2025-01-01')
            self.assertEqual(rate, 1.0)
        
        # Step 3: Test sector mapping
        with patch('modules.sector_mapper.DataFetcher') as MockDataFetcher:
            mock_fetcher = Mock()
            mock_fetcher.fetch_company_info.return_value = {'sector': 'Technology'}
            MockDataFetcher.return_value = mock_fetcher
            
            sector_mapper = SectorMapper(mock_fetcher)
            sector = sector_mapper.get_sector_for_ticker('005930')
            self.assertEqual(sector, 'Technology')
        
        # Step 4: Test portfolio calculations
        calculator = PortfolioCalculator()
        
        # Get data for calculation
        calculation_data = portfolio.get_data_for_calculation()
        self.assertIsNotNone(calculation_data)
        
        # Perform calculations
        results = calculator.calculate_all(calculation_data)
        
        # Check calculation results
        self.assertIsInstance(results, dict)
        self.assertIn('monthly_values', results)
        self.assertIn('returns', results)
        self.assertIn('risk_metrics', results)
        self.assertIn('allocation', results)
        
        # Check monthly values
        monthly_values = results['monthly_values']
        self.assertIn('dates', monthly_values)
        self.assertIn('values', monthly_values)
        self.assertEqual(len(monthly_values['dates']), 2)  # 2 months
        
        # Check returns
        returns = results['returns']
        self.assertIn('monthly_returns', returns)
        self.assertIn('total_return', returns)
        
        # Check allocation
        allocation = results['allocation']
        self.assertIn('by_sector', allocation)
        self.assertIn('top_holdings', allocation)
        
        # Step 5: Test report generation
        with patch('modules.report_gen.plt.savefig'), \
             patch('modules.report_gen.plt.figure'):
            
            # Mock config for report generator
            with patch.dict('modules.report_gen.__dict__', {
                'PROJECT_ROOT': self.temp_dir,
                'REPORTS_DIR': os.path.join(self.temp_dir, 'reports'),
                'CHARTS_DIR': os.path.join(self.temp_dir, 'charts'),
                'REPORT_TITLE': 'Test Report',
                'REPORT_AUTHOR': 'Test',
                'REPORT_VERSION': '1.0',
                'CURRENCY_SYMBOLS': {'KRW': '₩'},
                'BASE_CURRENCY': 'KRW'
            }):
                
                report_generator = ReportGenerator()
                
                # Generate report
                report_path = report_generator.generate_report(
                    portfolio_data=calculation_data,
                    calculation_results=results,
                    title="Test Portfolio Report",
                    include_charts=False
                )
                
                # Check report was generated
                self.assertIsNotNone(report_path)
                self.assertTrue(os.path.exists(report_path))
                self.assertTrue(report_path.endswith('.html'))
    
    def test_workflow_with_missing_data(self):
        """Test workflow with missing data that needs to be fetched."""
        # Create CSV with missing data
        missing_csv_data = pd.DataFrame({
            '기준일': ['2025-01-01', '2025-01-01'],
            '종목코드': ['005930', '000660'],
            '종목명': ['삼성전자', 'SK하이닉스'],
            '수량': [100, 50],
            '현재가': [None, None],  # Missing prices
            '통화': ['KRW', 'KRW'],
            '섹터': ['', '']  # Missing sectors
        })
        
        missing_csv_file = os.path.join(self.temp_dir, 'missing_data.csv')
        missing_csv_data.to_csv(missing_csv_file, index=False, encoding='utf-8-sig')
        
        # Load portfolio data
        portfolio = PortfolioData()
        success = portfolio.load_csv(missing_csv_file)
        
        self.assertTrue(success, "Failed to load CSV with missing data")
        
        # Check missing data identification
        missing_summary = portfolio.get_missing_data_summary()
        self.assertGreater(missing_summary['total_missing'], 0)
        self.assertGreater(len(portfolio.missing_data['prices']), 0)
        self.assertGreater(len(portfolio.missing_data['sectors']), 0)
        
        # Test data fetching for missing items
        with patch('modules.data_fetcher.yf.download') as mock_yf, \
             patch('modules.data_fetcher.requests.get') as mock_requests:
            
            # Mock successful fetches
            mock_yf.return_value = pd.DataFrame({
                'Close': [70000.0]
            }, index=[pd.Timestamp('2025-01-01')])
            
            mock_response = Mock()
            mock_response.json.return_value = {'rates': {'KRW': 1.0}}
            mock_response.raise_for_status.return_value = None
            mock_requests.return_value = mock_response
            
            data_fetcher = DataFetcher()
            
            # Fetch missing price
            for missing_item in portfolio.missing_data['prices']:
                price = data_fetcher.fetch_stock_price(
                    missing_item['ticker'],
                    missing_item['date']
                )
                self.assertIsNotNone(price)
                
                # Update portfolio with fetched price
                portfolio.update_missing_data('prices', [{
                    'ticker': missing_item['ticker'],
                    'date': missing_item['date'],
                    'value': price
                }])
        
        # Check that missing data was reduced
        updated_missing = portfolio.get_missing_data_summary()
        self.assertLess(updated_missing['total_missing'], missing_summary['total_missing'])
    
    def test_error_handling_in_workflow(self):
        """Test error handling at various stages of the workflow."""
        # Test 1: Invalid CSV file
        portfolio = PortfolioData()
        
        # Create invalid CSV (empty file)
        invalid_csv = os.path.join(self.temp_dir, 'invalid.csv')
        with open(invalid_csv, 'w', encoding='utf-8') as f:
            f.write('')
        
        success = portfolio.load_csv(invalid_csv)
        self.assertFalse(success, "Should fail to load invalid CSV")
        self.assertGreater(len(portfolio.data_quality_issues), 0)
        
        # Test 2: API failures in data fetching
        with patch('modules.data_fetcher.yf.download') as mock_yf:
            mock_yf.side_effect = Exception("API Error")
            
            data_fetcher = DataFetcher()
            price = data_fetcher.fetch_stock_price('INVALID', '2025-01-01')
            
            self.assertIsNone(price, "Should return None on API failure")
        
        # Test 3: Empty data for calculations
        calculator = PortfolioCalculator()
        empty_results = calculator.calculate_all(pd.DataFrame())
        
        self.assertEqual(empty_results, {}, "Should return empty dict for empty data")
        
        # Test 4: Report generation with invalid data
        with patch.dict('modules.report_gen.__dict__', {
            'PROJECT_ROOT': self.temp_dir,
            'REPORTS_DIR': os.path.join(self.temp_dir, 'reports'),
            'CHARTS_DIR': os.path.join(self.temp_dir, 'charts'),
            'REPORT_TITLE': 'Test',
            'REPORT_AUTHOR': 'Test',
            'REPORT_VERSION': '1.0',
            'CURRENCY_SYMBOLS': {},
            'BASE_CURRENCY': 'KRW'
        }):
            
            report_generator = ReportGenerator()
            
            # Try to generate report with None data
            report_path = report_generator.generate_report(
                portfolio_data=None,
                calculation_results=None,
                title="Invalid Report"
            )
            
            self.assertIsNone(report_path, "Should return None for invalid data")
    
    def test_data_persistence(self):
        """Test saving and loading of processed data."""
        # Load and process data
        portfolio = PortfolioData()
        success = portfolio.load_csv(self.csv_file)
        self.assertTrue(success)
        
        # Save processed data
        save_path = os.path.join(self.temp_dir, 'processed_data.csv')
        save_success = portfolio.save_to_original_format(save_path)
        
        self.assertTrue(save_success, "Failed to save processed data")
        self.assertTrue(os.path.exists(save_path))
        
        # Load saved data and verify
        saved_data = pd.read_csv(save_path, encoding='utf-8-sig')
        self.assertEqual(len(saved_data), len(portfolio.raw_data))
        
        # Check that essential columns are preserved
        expected_columns = ['기준일', '종목코드', '종목명', '수량', '현재가', '통화', '섹터']
        for col in expected_columns:
            self.assertIn(col, saved_data.columns)
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        # Create larger dataset (1000 rows)
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS')
        tickers = [f'TICKER{i:03d}' for i in range(20)]  # 20 unique tickers
        
        large_data = []
        for date in dates:
            for ticker in tickers:
                large_data.append({
                    '기준일': date.strftime('%Y-%m-%d'),
                    '종목코드': ticker,
                    '종목명': f'Company {ticker}',
                    '수량': 100,
                    '현재가': 10000.0,
                    '통화': 'KRW',
                    '섹터': 'Technology'
                })
        
        large_df = pd.DataFrame(large_data)
        large_csv_file = os.path.join(self.temp_dir, 'large_portfolio.csv')
        large_df.to_csv(large_csv_file, index=False, encoding='utf-8-sig')
        
        # Load large dataset
        portfolio = PortfolioData()
        
        import time
        start_time = time.time()
        success = portfolio.load_csv(large_csv_file)
        load_time = time.time() - start_time
        
        self.assertTrue(success, "Failed to load large CSV")
        self.assertEqual(len(portfolio.processed_data), len(large_data))
        
        # Performance check: loading should complete in reasonable time
        self.assertLess(load_time, 5.0, f"Loading took too long: {load_time:.2f} seconds")
        
        # Test calculations with large dataset
        calculator = PortfolioCalculator()
        
        start_time = time.time()
        calculation_data = portfolio.get_data_for_calculation()
        results = calculator.calculate_all(calculation_data)
        calc_time = time.time() - start_time
        
        self.assertIsInstance(results, dict)
        self.assertIn('monthly_values', results)
        
        # Performance check: calculations should complete in reasonable time
        self.assertLess(calc_time, 10.0, f"Calculations took too long: {calc_time:.2f} seconds")


if __name__ == '__main__':
    unittest.main()