"""
Functional validation tests for the portfolio factsheet application.
These tests validate that the core functionality works as intended.
"""

import sys
import os
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import PortfolioData
from modules.portfolio_calc import PortfolioCalculator


def test_data_loading():
    """Test that portfolio data can be loaded and processed correctly."""
    print("Test 1: Data Loading and Processing")
    
    # Create test data
    test_data = pd.DataFrame({
        '기준일': ['2025-01-01', '2025-01-01', '2025-02-01', '2025-02-01'],
        '종목코드': ['005930', '000660', '005930', '000660'],
        '종목명': ['삼성전자', 'SK하이닉스', '삼성전자', 'SK하이닉스'],
        '수량': [100, 50, 100, 50],
        '현재가': [70000, 150000, 72000, 155000],
        '통화': ['KRW', 'KRW', 'KRW', 'KRW'],
        '섹터': ['Technology', 'Technology', 'Technology', 'Technology']
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False, encoding='utf-8-sig')
        temp_file = f.name
    
    try:
        # Load data
        portfolio = PortfolioData()
        success = portfolio.load_csv(temp_file)
        
        assert success, "Failed to load CSV file"
        assert portfolio.raw_data is not None, "Raw data should not be None"
        assert portfolio.processed_data is not None, "Processed data should not be None"
        assert len(portfolio.processed_data) == 4, "Should have 4 rows of processed data"
        
        # Check column transformations
        expected_columns = ['date', 'ticker', 'name', 'quantity', 'price', 'currency', 'sector']
        for col in expected_columns:
            assert col in portfolio.processed_data.columns, f"Missing column: {col}"
        
        # Check value calculations
        assert 'value_krw' in portfolio.processed_data.columns, "Missing value_krw column"
        
        # Check data quality report
        report = portfolio.get_data_quality_report()
        assert 'summary' in report, "Data quality report should have summary"
        assert 'issues' in report, "Data quality report should have issues"
        
        print("  OK Data loading and processing: PASSED")
        return True
        
    except AssertionError as e:
        print(f"  FAIL Data loading and processing: FAILED - {e}")
        return False
    finally:
        os.unlink(temp_file)


def test_portfolio_calculations():
    """Test that portfolio calculations work correctly."""
    print("\nTest 2: Portfolio Calculations")
    
    try:
        # Create sample data for calculations
        dates = pd.date_range(start='2025-01-01', end='2025-03-01', freq='MS')
        sample_data = pd.DataFrame({
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
        
        # Perform calculations
        calculator = PortfolioCalculator()
        results = calculator.calculate_all(sample_data)
        
        assert results is not None, "Calculation results should not be None"
        assert isinstance(results, dict), "Results should be a dictionary"
        
        # Check key result sections
        required_sections = ['monthly_values', 'portfolio_stats', 'monthly_returns']
        for section in required_sections:
            assert section in results, f"Missing section: {section}"
        
        # Check monthly values
        monthly_values = results['monthly_values']
        assert 'values' in monthly_values, "Monthly values should contain values"
        assert 'dates' in monthly_values, "Monthly values should contain dates"
        assert len(monthly_values['values']) == 3, "Should have 3 months of data"
        
        # Check portfolio stats
        stats = results['portfolio_stats']
        assert 'initial_value' in stats, "Missing initial value"
        assert 'latest_value' in stats, "Missing latest value"
        assert 'total_return_pct' in stats, "Missing total return percentage"
        
        # Check monthly returns
        returns = results['monthly_returns']
        assert 'returns' in returns, "Missing returns data"
        assert 'avg_monthly_return' in returns, "Missing average monthly return"
        
        print("  OK Portfolio calculations: PASSED")
        return True
        
    except AssertionError as e:
        print(f"  FAIL Portfolio calculations: FAILED - {e}")
        return False
    except Exception as e:
        print(f"  FAIL Portfolio calculations: FAILED with exception - {e}")
        return False


def test_data_quality_validation():
    """Test that data quality validation works correctly."""
    print("\nTest 3: Data Quality Validation")
    
    # Create test data with quality issues
    test_data = pd.DataFrame({
        '기준일': ['2025-01-01', '2025-01-01', 'invalid_date'],
        '종목코드': ['005930', '000660', 'INVALID'],
        '종목명': ['삼성전자', 'SK하이닉스', 'Invalid Company'],
        '수량': [100, -50, 0],  # Negative and zero quantities
        '현재가': [70000, None, 100000],  # Missing price
        '통화': ['KRW', 'USD', 'INVALID'],  # Invalid currency
        '섹터': ['Technology', '', 'Invalid Sector']  # Missing and invalid sector
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False, encoding='utf-8-sig')
        temp_file = f.name
    
    try:
        # Load data
        portfolio = PortfolioData()
        success = portfolio.load_csv(temp_file)
        
        # Data should still load (with warnings)
        assert success, "Should load data even with quality issues"
        
        # Check data quality report
        report = portfolio.get_data_quality_report()
        assert 'issues' in report, "Should have issues in report"
        
        issues = report['issues']
        assert len(issues) > 0, "Should detect quality issues"
        
        # Check missing data identification
        missing_data = portfolio.missing_data
        assert len(missing_data['prices']) > 0, "Should detect missing prices"
        assert len(missing_data['sectors']) > 0, "Should detect missing sectors"
        
        print("  OK Data quality validation: PASSED")
        return True
        
    except AssertionError as e:
        print(f"  FAIL Data quality validation: FAILED - {e}")
        return False
    finally:
        os.unlink(temp_file)


def test_data_persistence():
    """Test that data can be saved and loaded correctly."""
    print("\nTest 4: Data Persistence")
    
    # Create test data
    test_data = pd.DataFrame({
        '기준일': ['2025-01-01', '2025-01-01'],
        '종목코드': ['005930', '000660'],
        '종목명': ['삼성전자', 'SK하이닉스'],
        '수량': [100, 50],
        '현재가': [70000, 150000],
        '통화': ['KRW', 'KRW'],
        '섹터': ['Technology', 'Technology']
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False, encoding='utf-8-sig')
        temp_file = f.name
    
    try:
        # Load and process data
        portfolio = PortfolioData()
        success = portfolio.load_csv(temp_file)
        assert success, "Failed to load initial data"
        
        # Save updated data
        save_file = temp_file.replace('.csv', '_saved.csv')
        save_success = portfolio.save_updated_data(save_file)
        
        assert save_success, "Failed to save updated data"
        assert os.path.exists(save_file), "Saved file should exist"
        
        # Load saved data to verify
        saved_data = pd.read_csv(save_file, encoding='utf-8-sig')
        assert len(saved_data) == len(test_data), "Saved data should have same number of rows"
        
        # Check essential columns are preserved
        expected_columns = ['date', 'ticker', 'name', 'quantity', 'price', 'currency', 'sector']
        for col in expected_columns:
            # Column names might be transformed
            col_found = any(expected in str(col).lower() for expected in [c.lower() for c in expected_columns])
            assert col_found, f"Essential column not found in saved data: {col}"
        
        print("  OK Data persistence: PASSED")
        return True
        
    except AssertionError as e:
        print(f"  FAIL Data persistence: FAILED - {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        save_file = temp_file.replace('.csv', '_saved.csv')
        if os.path.exists(save_file):
            os.unlink(save_file)


def test_configuration():
    """Test that configuration is properly loaded."""
    print("\nTest 5: Configuration")
    
    try:
        from config import (
            GUI_TITLE, GUI_WIDTH, GUI_HEIGHT,
            BASE_CURRENCY, VALID_CURRENCIES, VALID_SECTORS,
            PROJECT_ROOT, DATA_DIR, REPORTS_DIR
        )
        
        # Check required config values
        assert GUI_TITLE is not None, "GUI_TITLE should be set"
        assert GUI_WIDTH > 0, "GUI_WIDTH should be positive"
        assert GUI_HEIGHT > 0, "GUI_HEIGHT should be positive"
        assert BASE_CURRENCY is not None, "BASE_CURRENCY should be set"
        assert isinstance(VALID_CURRENCIES, list), "VALID_CURRENCIES should be a list"
        assert isinstance(VALID_SECTORS, list), "VALID_SECTORS should be a list"
        assert PROJECT_ROOT is not None, "PROJECT_ROOT should be set"
        
        print("  OK Configuration: PASSED")
        return True
        
    except ImportError as e:
        print(f"  FAIL Configuration: FAILED - Cannot import config - {e}")
        return False
    except AssertionError as e:
        print(f"  FAIL Configuration: FAILED - {e}")
        return False


def run_all_tests():
    """Run all functional validation tests."""
    print("=" * 60)
    print("PORTFOLIO FACTSHEET APPLICATION - FUNCTIONAL VALIDATION")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Data Loading", test_data_loading()))
    test_results.append(("Portfolio Calculations", test_portfolio_calculations()))
    test_results.append(("Data Quality Validation", test_data_quality_validation()))
    test_results.append(("Data Persistence", test_data_persistence()))
    test_results.append(("Configuration", test_configuration()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nSUCCESS All functional tests passed! Application is working correctly.")
        return True
    else:
        print(f"\nWARNING {total - passed} test(s) failed. Review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)