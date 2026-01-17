"""
Unit tests for portfolio_calc module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from modules.portfolio_calc import PortfolioCalculator


class TestPortfolioCalculator(unittest.TestCase):
    """Test cases for PortfolioCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        self.calculator = PortfolioCalculator(base_currency="KRW")
        
        # Create sample portfolio data
        dates = pd.date_range(start='2025-01-01', end='2025-06-01', freq='MS')
        self.sample_data = pd.DataFrame({
            'year_month': np.repeat(dates, 3),
            'ticker': ['005930', '000660', 'AAPL'] * len(dates),
            'quantity': [100, 50, 30] * len(dates),
            'price': [70000, 150000, 180] * len(dates),
            'exchange_rate': [1.0, 1.0, 1300.0] * len(dates),  # 1 USD = 1300 KRW
            'sector': ['Technology', 'Technology', 'Technology'] * len(dates),
            'currency': ['KRW', 'KRW', 'USD'] * len(dates)
        })
        
        # Add some price variation
        for i in range(len(self.sample_data)):
            if self.sample_data.loc[i, 'ticker'] == '005930':
                self.sample_data.loc[i, 'price'] = 70000 + i * 1000
            elif self.sample_data.loc[i, 'ticker'] == '000660':
                self.sample_data.loc[i, 'price'] = 150000 + i * 2000
            else:  # AAPL
                self.sample_data.loc[i, 'price'] = 180 + i * 5
    
    def test_calculate_all(self):
        """Test comprehensive calculation of all metrics."""
        results = self.calculator.calculate_all(self.sample_data)
        
        # Check that all expected result sections exist
        expected_sections = [
            'monthly_values',
            'returns',
            'risk_metrics',
            'allocation',
            'benchmark_comparison',
            'performance_attribution',
            'analytics'
        ]
        
        for section in expected_sections:
            self.assertIn(section, results)
    
    def test_calculate_portfolio_values(self):
        """Test calculation of portfolio values."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_portfolio_values()
        
        # Check monthly values were calculated
        self.assertIn('monthly_values', self.calculator.results)
        monthly_values = self.calculator.results['monthly_values']
        
        self.assertIn('dates', monthly_values)
        self.assertIn('values', monthly_values)
        self.assertIn('currency', monthly_values)
        
        # Should have 6 months of data
        self.assertEqual(len(monthly_values['dates']), 6)
        self.assertEqual(len(monthly_values['values']), 6)
        
        # Check that values are positive
        for value in monthly_values['values']:
            self.assertGreater(value, 0)
    
    def test_calculate_monthly_returns(self):
        """Test calculation of monthly returns."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_portfolio_values()
        self.calculator._calculate_monthly_returns()
        
        # Check returns were calculated
        self.assertIn('returns', self.calculator.results)
        returns = self.calculator.results['returns']
        
        self.assertIn('monthly_returns', returns)
        self.assertIn('annualized_return', returns)
        self.assertIn('total_return', returns)
        
        # Monthly returns should have one less element than monthly values
        monthly_values = self.calculator.results['monthly_values']['values']
        monthly_returns = returns['monthly_returns']
        self.assertEqual(len(monthly_returns), len(monthly_values) - 1)
    
    def test_calculate_cumulative_returns(self):
        """Test calculation of cumulative returns."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_portfolio_values()
        self.calculator._calculate_monthly_returns()
        self.calculator._calculate_cumulative_returns()
        
        # Check cumulative returns were calculated
        self.assertIn('returns', self.calculator.results)
        returns = self.calculator.results['returns']
        
        self.assertIn('cumulative_returns', returns)
        self.assertIn('cumulative_return', returns)
        
        # Cumulative returns should have same length as monthly values
        monthly_values = self.calculator.results['monthly_values']['values']
        cumulative_returns = returns['cumulative_returns']
        self.assertEqual(len(cumulative_returns), len(monthly_values))
        
        # First cumulative return should be 0 (starting point)
        self.assertEqual(cumulative_returns[0], 0)
    
    def test_calculate_risk_metrics(self):
        """Test calculation of risk metrics."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_portfolio_values()
        self.calculator._calculate_monthly_returns()
        self.calculator._calculate_risk_metrics()
        
        # Check risk metrics were calculated
        self.assertIn('risk_metrics', self.calculator.results)
        risk_metrics = self.calculator.results['risk_metrics']
        
        expected_metrics = [
            'volatility',
            'sharpe_ratio',
            'max_drawdown',
            'max_drawdown_period',
            'var_95',
            'cvar_95'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics)
        
        # Volatility should be positive
        self.assertGreaterEqual(risk_metrics['volatility'], 0)
        
        # Max drawdown should be negative or zero
        self.assertLessEqual(risk_metrics['max_drawdown'], 0)
    
    def test_calculate_allocation_metrics(self):
        """Test calculation of allocation metrics."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_allocation_metrics()
        
        # Check allocation metrics were calculated
        self.assertIn('allocation', self.calculator.results)
        allocation = self.calculator.results['allocation']
        
        expected_sections = [
            'by_sector',
            'by_currency',
            'top_holdings',
            'concentration'
        ]
        
        for section in expected_sections:
            self.assertIn(section, allocation)
        
        # Check sector allocation
        sector_allocation = allocation['by_sector']
        self.assertIsInstance(sector_allocation, list)
        
        # All sectors should sum to approximately 100%
        total_sector_pct = sum(item['percentage'] for item in sector_allocation)
        self.assertAlmostEqual(total_sector_pct, 100.0, delta=0.1)
    
    def test_calculate_benchmark_comparisons(self):
        """Test calculation of benchmark comparisons."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_portfolio_values()
        self.calculator._calculate_monthly_returns()
        self.calculator._calculate_benchmark_comparisons()
        
        # Check benchmark comparisons were calculated
        self.assertIn('benchmark_comparison', self.calculator.results)
        benchmark_comparison = self.calculator.results['benchmark_comparison']
        
        self.assertIn('benchmarks', benchmark_comparison)
        self.assertIn('outperformance', benchmark_comparison)
        
        # Should have benchmark data
        benchmarks = benchmark_comparison['benchmarks']
        self.assertIsInstance(benchmarks, dict)
        self.assertGreater(len(benchmarks), 0)
    
    def test_calculate_performance_attribution(self):
        """Test calculation of performance attribution."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_portfolio_values()
        self.calculator._calculate_monthly_returns()
        self.calculator._calculate_performance_attribution()
        
        # Check performance attribution was calculated
        self.assertIn('performance_attribution', self.calculator.results)
        attribution = self.calculator.results['performance_attribution']
        
        expected_sections = [
            'by_sector',
            'by_currency',
            'top_contributors',
            'top_detractors'
        ]
        
        for section in expected_sections:
            self.assertIn(section, attribution)
    
    def test_calculate_additional_analytics(self):
        """Test calculation of additional analytics."""
        self.calculator.portfolio_data = self.sample_data.copy()
        self.calculator._calculate_additional_analytics()
        
        # Check additional analytics were calculated
        self.assertIn('analytics', self.calculator.results)
        analytics = self.calculator.results['analytics']
        
        expected_metrics = [
            'turnover_rate',
            'active_share',
            'tracking_error',
            'information_ratio'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, analytics)
    
    def test_empty_data(self):
        """Test calculation with empty data."""
        empty_data = pd.DataFrame(columns=self.sample_data.columns)
        results = self.calculator.calculate_all(empty_data)
        
        # Should return empty dictionary
        self.assertEqual(results, {})
    
    def test_single_month_data(self):
        """Test calculation with only one month of data."""
        single_month_data = self.sample_data[self.sample_data['year_month'] == '2025-01-01'].copy()
        results = self.calculator.calculate_all(single_month_data)
        
        # Should still calculate basic metrics
        self.assertIn('monthly_values', results)
        self.assertIn('allocation', results)
        
        # Returns and risk metrics may not be calculable with single data point
        returns = results.get('returns', {})
        if returns:
            self.assertIn('total_return', returns)
            self.assertEqual(returns['total_return'], 0)  # No return with single point
    
    def test_currency_conversion(self):
        """Test proper currency conversion in calculations."""
        # Create data with mixed currencies
        mixed_currency_data = pd.DataFrame({
            'year_month': ['2025-01-01', '2025-01-01', '2025-02-01', '2025-02-01'],
            'ticker': ['005930', 'AAPL', '005930', 'AAPL'],
            'quantity': [100, 30, 100, 30],
            'price': [70000, 180, 72000, 185],
            'exchange_rate': [1.0, 1300.0, 1.0, 1320.0],  # USD appreciation
            'sector': ['Technology', 'Technology', 'Technology', 'Technology'],
            'currency': ['KRW', 'USD', 'KRW', 'USD']
        })
        
        results = self.calculator.calculate_all(mixed_currency_data)
        
        # Check that values were properly converted to KRW
        monthly_values = results['monthly_values']['values']
        self.assertEqual(len(monthly_values), 2)  # Two months
        
        # Second month should have higher value due to USD appreciation
        self.assertGreater(monthly_values[1], monthly_values[0])
    
    def test_negative_returns(self):
        """Test calculation with negative returns."""
        # Create data with declining prices
        declining_data = pd.DataFrame({
            'year_month': ['2025-01-01', '2025-02-01', '2025-03-01'],
            'ticker': ['005930', '005930', '005930'],
            'quantity': [100, 100, 100],
            'price': [70000, 65000, 60000],  # Declining prices
            'exchange_rate': [1.0, 1.0, 1.0],
            'sector': ['Technology', 'Technology', 'Technology'],
            'currency': ['KRW', 'KRW', 'KRW']
        })
        
        results = self.calculator.calculate_all(declining_data)
        
        # Check returns are negative
        returns = results['returns']
        monthly_returns = returns['monthly_returns']
        
        for ret in monthly_returns:
            self.assertLess(ret, 0)  # All returns should be negative
        
        # Total return should be negative
        self.assertLess(returns['total_return'], 0)
        
        # Max drawdown should be significant
        risk_metrics = results['risk_metrics']
        self.assertLess(risk_metrics['max_drawdown'], -0.1)  # At least 10% drawdown


if __name__ == '__main__':
    unittest.main()