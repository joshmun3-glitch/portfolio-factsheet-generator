"""
Unit tests for report_gen module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt

from modules.report_gen import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class."""
    
    def setUp(self):
        """Set up test data."""
        # Create temp directory for reports
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch config constants
        self.config_patcher = patch.dict('modules.report_gen.__dict__', {
            'PROJECT_ROOT': self.temp_dir,
            'REPORTS_DIR': os.path.join(self.temp_dir, 'reports'),
            'CHARTS_DIR': os.path.join(self.temp_dir, 'charts'),
            'REPORT_TITLE': 'Test Portfolio Report',
            'REPORT_AUTHOR': 'Test Author',
            'REPORT_VERSION': '1.0.0',
            'CURRENCY_SYMBOLS': {'KRW': '₩', 'USD': '$', 'EUR': '€'},
            'BASE_CURRENCY': 'KRW'
        })
        self.config_patcher.start()
        
        self.report_generator = ReportGenerator()
        
        # Create sample portfolio data
        dates = pd.date_range(start='2025-01-01', end='2025-03-01', freq='MS')
        self.sample_portfolio_data = pd.DataFrame({
            'year_month': np.repeat(dates, 2),
            'ticker': ['005930', '000660'] * len(dates),
            'name': ['삼성전자', 'SK하이닉스'] * len(dates),
            'quantity': [100, 50] * len(dates),
            'price': [70000, 150000, 72000, 155000, 75000, 160000],
            'exchange_rate': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'sector': ['Technology', 'Technology'] * len(dates),
            'currency': ['KRW', 'KRW'] * len(dates),
            'value_krw': [7000000, 7500000, 7200000, 7750000, 7500000, 8000000]
        })
        
        # Create sample calculation results
        self.sample_calculation_results = {
            'monthly_values': {
                'dates': ['2025-01', '2025-02', '2025-03'],
                'values': [14500000, 14950000, 15500000],
                'currency': 'KRW'
            },
            'returns': {
                'monthly_returns': [0.0310, 0.0368],
                'annualized_return': 0.248,
                'total_return': 0.0690,
                'cumulative_returns': [0, 0.0310, 0.0690],
                'cumulative_return': 0.0690
            },
            'risk_metrics': {
                'volatility': 0.152,
                'sharpe_ratio': 1.63,
                'max_drawdown': -0.082,
                'max_drawdown_period': '2025-01 to 2025-02',
                'var_95': -0.125,
                'cvar_95': -0.158
            },
            'allocation': {
                'by_sector': [
                    {'sector': 'Technology', 'percentage': 100.0, 'value': 15500000}
                ],
                'by_currency': [
                    {'currency': 'KRW', 'percentage': 100.0, 'value': 15500000}
                ],
                'top_holdings': [
                    {'ticker': '005930', 'name': '삼성전자', 'percentage': 48.4, 'value': 7500000},
                    {'ticker': '000660', 'name': 'SK하이닉스', 'percentage': 51.6, 'value': 8000000}
                ],
                'concentration': {
                    'top_3_concentration': 100.0,
                    'herfindahl_index': 0.5008
                }
            },
            'benchmark_comparison': {
                'benchmarks': {
                    'KOSPI': {
                        'returns': [0.025, 0.030],
                        'total_return': 0.056,
                        'volatility': 0.145
                    }
                },
                'outperformance': {
                    'vs_kospi': 0.013
                }
            },
            'performance_attribution': {
                'by_sector': [
                    {'sector': 'Technology', 'contribution': 0.0690}
                ],
                'by_currency': [
                    {'currency': 'KRW', 'contribution': 0.0690}
                ],
                'top_contributors': [
                    {'ticker': '000660', 'name': 'SK하이닉스', 'contribution': 0.0350},
                    {'ticker': '005930', 'name': '삼성전자', 'contribution': 0.0340}
                ],
                'top_detractors': []
            },
            'analytics': {
                'turnover_rate': 0.15,
                'active_share': 0.85,
                'tracking_error': 0.032,
                'information_ratio': 0.41
            }
        }
    
    def tearDown(self):
        """Clean up test files."""
        self.config_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ReportGenerator initialization."""
        # Check that directories were created
        self.assertTrue(os.path.exists(self.report_generator.reports_dir))
        self.assertTrue(os.path.exists(self.report_generator.charts_dir))
        
        # Check matplotlib backend
        self.assertEqual(plt.get_backend(), 'Agg')
    
    def test_prepare_template_data(self):
        """Test preparation of template data."""
        template_data = self.report_generator._prepare_template_data(
            self.sample_portfolio_data,
            self.sample_calculation_results,
            title="Custom Title",
            include_charts=True
        )
        
        # Check basic report info
        self.assertEqual(template_data["title"], "Custom Title")
        self.assertEqual(template_data["author"], "Test Author")
        self.assertEqual(template_data["version"], "1.0.0")
        self.assertIn("generation_date", template_data)
        
        # Check portfolio summary
        self.assertIn("portfolio_summary", template_data)
        summary = template_data["portfolio_summary"]
        self.assertIn("period", summary)
        self.assertIn("total_value", summary)
        self.assertIn("currency", summary)
        
        # Check calculation results were included
        self.assertIn("returns", template_data)
        self.assertIn("risk_metrics", template_data)
        self.assertIn("allocation", template_data)
        self.assertIn("benchmark_comparison", template_data)
        self.assertIn("performance_attribution", template_data)
        self.assertIn("analytics", template_data)
    
    @patch('modules.report_gen.plt.savefig')
    @patch('modules.report_gen.plt.figure')
    def test_generate_charts(self, mock_figure, mock_savefig):
        """Test chart generation."""
        # Mock figure and axes
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        template_data = self.report_generator._prepare_template_data(
            self.sample_portfolio_data,
            self.sample_calculation_results,
            include_charts=True
        )
        
        charts = self.report_generator._generate_charts(template_data, self.sample_calculation_results)
        
        # Check that charts were generated
        self.assertIsInstance(charts, dict)
        
        # Check expected chart types
        expected_charts = [
            'performance_chart',
            'allocation_chart',
            'risk_chart',
            'returns_distribution'
        ]
        
        for chart_type in expected_charts:
            self.assertIn(chart_type, charts)
            
            # Check chart data structure
            chart_data = charts[chart_type]
            self.assertIn('type', chart_data)
            self.assertIn('data', chart_data)
            self.assertIn('title', chart_data)
    
    def test_generate_html_content(self):
        """Test HTML content generation."""
        template_data = self.report_generator._prepare_template_data(
            self.sample_portfolio_data,
            self.sample_calculation_results,
            include_charts=False
        )
        
        html_content = self.report_generator._generate_html_content(template_data)
        
        # Check HTML structure
        self.assertIsInstance(html_content, str)
        self.assertGreater(len(html_content), 0)
        
        # Check for essential HTML elements
        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn('<html', html_content)
        self.assertIn('<head>', html_content)
        self.assertIn('<body>', html_content)
        self.assertIn('</html>', html_content)
        
        # Check for report title
        self.assertIn('Test Portfolio Report', html_content)
        
        # Check for portfolio data
        self.assertIn('삼성전자', html_content)
        self.assertIn('SK하이닉스', html_content)
    
    @patch('modules.report_gen.ReportGenerator._generate_charts')
    @patch('modules.report_gen.ReportGenerator._generate_html_content')
    def test_generate_report(self, mock_generate_html, mock_generate_charts):
        """Test complete report generation."""
        # Mock chart generation
        mock_generate_charts.return_value = {
            'performance_chart': {'type': 'line', 'data': {}, 'title': 'Performance'}
        }
        
        # Mock HTML generation
        mock_generate_html.return_value = '<html><body>Test Report</body></html>'
        
        # Generate report
        report_path = self.report_generator.generate_report(
            portfolio_data=self.sample_portfolio_data,
            calculation_results=self.sample_calculation_results,
            title="Test Report",
            include_charts=True
        )
        
        # Check report was generated
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
        
        # Check file extension
        self.assertTrue(report_path.endswith('.html'))
        
        # Check file content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertEqual(content, '<html><body>Test Report</body></html>')
    
    def test_generate_report_no_charts(self):
        """Test report generation without charts."""
        report_path = self.report_generator.generate_report(
            portfolio_data=self.sample_portfolio_data,
            calculation_results=self.sample_calculation_results,
            title="Test Report No Charts",
            include_charts=False
        )
        
        # Check report was generated
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
    
    def test_generate_report_empty_data(self):
        """Test report generation with empty data."""
        empty_portfolio_data = pd.DataFrame()
        empty_calculation_results = {}
        
        report_path = self.report_generator.generate_report(
            portfolio_data=empty_portfolio_data,
            calculation_results=empty_calculation_results,
            title="Empty Report"
        )
        
        # Should still generate report (but may be empty)
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
    
    def test_format_currency(self):
        """Test currency formatting."""
        # Test KRW formatting
        formatted_krw = self.report_generator._format_currency(1000000, 'KRW')
        self.assertIn('₩', formatted_krw)
        self.assertIn('1,000,000', formatted_krw)
        
        # Test USD formatting
        formatted_usd = self.report_generator._format_currency(1000, 'USD')
        self.assertIn('$', formatted_usd)
        self.assertIn('1,000', formatted_usd)
        
        # Test unknown currency
        formatted_unknown = self.report_generator._format_currency(500, 'GBP')
        self.assertIn('500', formatted_unknown)  # Should still format number
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        # Test positive percentage
        formatted_positive = self.report_generator._format_percentage(0.15678)
        self.assertEqual(formatted_positive, "15.68%")
        
        # Test negative percentage
        formatted_negative = self.report_generator._format_percentage(-0.08234)
        self.assertEqual(formatted_negative, "-8.23%")
        
        # Test zero percentage
        formatted_zero = self.report_generator._format_percentage(0.0)
        self.assertEqual(formatted_zero, "0.00%")
        
        # Test very small percentage
        formatted_small = self.report_generator._format_percentage(0.000123)
        self.assertEqual(formatted_small, "0.01%")
    
    def test_format_date(self):
        """Test date formatting."""
        # Test string date
        formatted_string = self.report_generator._format_date('2025-01-15')
        self.assertEqual(formatted_string, "2025-01-15")
        
        # Test datetime object
        dt = datetime(2025, 1, 15)
        formatted_dt = self.report_generator._format_date(dt)
        self.assertEqual(formatted_dt, "2025-01-15")
        
        # Test pandas Timestamp
        ts = pd.Timestamp('2025-01-15')
        formatted_ts = self.report_generator._format_date(ts)
        self.assertEqual(formatted_ts, "2025-01-15")
    
    def test_get_color_for_performance(self):
        """Test color coding for performance values."""
        # Test positive return (green)
        color_positive = self.report_generator._get_color_for_performance(0.15)
        self.assertEqual(color_positive, "green")
        
        # Test negative return (red)
        color_negative = self.report_generator._get_color_for_performance(-0.08)
        self.assertEqual(color_negative, "red")
        
        # Test zero return (black)
        color_zero = self.report_generator._get_color_for_performance(0.0)
        self.assertEqual(color_zero, "black")
        
        # Test small positive (green)
        color_small_positive = self.report_generator._get_color_for_performance(0.001)
        self.assertEqual(color_small_positive, "green")
        
        # Test small negative (red)
        color_small_negative = self.report_generator._get_color_for_performance(-0.001)
        self.assertEqual(color_small_negative, "red")


if __name__ == '__main__':
    unittest.main()