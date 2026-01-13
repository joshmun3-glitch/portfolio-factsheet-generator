"""
Data loader module for parsing portfolio CSV files with Korean encoding support.
Enhanced with comprehensive data quality validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from typing import Dict, List, Optional, Any

from config import (
    BASE_CURRENCY, VALID_CURRENCIES, VALID_SECTORS,
    MIN_PORTFOLIO_VALUE, MAX_PRICE_CHANGE_PCT
)

# Set up logging
logger = logging.getLogger(__name__)


class PortfolioData:
    """Class to load and manage portfolio data from CSV files."""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.missing_data = {
            "prices": [],
            "exchange_rates": [],
            "sectors": [],
            "weights": []
        }
        self.data_quality_issues = []
        self.monthly_dates = []
        self.portfolio_summary = {}
        
    def load_csv(self, filepath: str) -> bool:
        """Load portfolio data from CSV file."""
        try:
            # Try different encodings for Korean support
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    self.raw_data = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error("Failed to load CSV with any encoding")
                return False
            
            # Validate basic structure
            if not self._validate_csv_structure():
                return False
            
            # Process the data
            self._process_raw_data()
            
            # Validate data quality
            self._validate_data_quality()
            
            # Identify missing data
            self._identify_missing_data()
            
            # Calculate portfolio summary
            self._calculate_portfolio_summary()
            
            logger.info(f"Successfully loaded portfolio data with {len(self.raw_data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            self.data_quality_issues.append(f"Error loading CSV: {str(e)}")
            return False
    
    def _validate_csv_structure(self) -> bool:
        """Validate the CSV file has required columns and structure."""
        if self.raw_data is None:
            return False
            
        required_columns = ['기준일', '종목코드', '종목명', '수량', '현재가']
        
        # Check required columns
        missing_columns = []
        for col in required_columns:
            if col not in self.raw_data.columns:
                missing_columns.append(col)
                
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            self.data_quality_issues.append(f"Missing columns: {missing_columns}")
            return False
        
        # Check for empty data
        if self.raw_data.empty:
            logger.error("CSV file is empty")
            self.data_quality_issues.append("CSV file is empty")
            return False
        
        # Check date format
        try:
            self.raw_data['기준일'] = pd.to_datetime(self.raw_data['기준일'])
        except Exception as e:
            logger.error(f"Invalid date format: {e}")
            self.data_quality_issues.append(f"Invalid date format: {str(e)}")
            return False
        
        # Check numeric columns
        numeric_columns = ['수량', '현재가']
        for col in numeric_columns:
            if col in self.raw_data.columns:
                try:
                    self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
                except Exception as e:
                    logger.error(f"Invalid numeric format for {col}: {e}")
                    self.data_quality_issues.append(f"Invalid numeric format for {col}: {str(e)}")
        
        return True
    
    def _process_raw_data(self):
        """Process raw CSV data into standardized format."""
        if self.raw_data is None:
            return
        
        df = self.raw_data.copy()
        
        # Standardize column names
        column_mapping = {
            '기준일': 'date',
            '종목코드': 'ticker',
            '종목명': 'name',
            '수량': 'quantity',
            '현재가': 'price',
            '통화': 'currency',
            '섹터': 'sector',
            '환율': 'exchange_rate'
        }
        
        # Rename columns that exist
        for korean, english in column_mapping.items():
            if korean in df.columns:
                df.rename(columns={korean: english}, inplace=True)
        
        # Set default values for missing columns
        if 'currency' not in df.columns:
            df['currency'] = 'KRW'
        
        if 'exchange_rate' not in df.columns:
            df['exchange_rate'] = 1.0
        
        # Calculate value in local currency
        df['value_local'] = df['quantity'] * df['price']
        
        # Calculate value in KRW
        df['value_krw'] = df['value_local'] * df['exchange_rate']
        
        # Extract unique dates
        self.monthly_dates = sorted(df['date'].dt.strftime('%Y-%m').unique())
        
        self.processed_data = df
        
        logger.info(f"Processed data: {len(df)} rows, {len(self.monthly_dates)} unique months")
    
    def _validate_data_quality(self):
        """Validate data quality with comprehensive checks."""
        if self.processed_data is None:
            logger.error("No processed data available")
            return
        
        df = self.processed_data
        logger.info("Starting comprehensive data quality validation...")
        
        # 1. Check for duplicate entries
        if 'date' in df.columns and 'ticker' in df.columns:
            duplicate_check = df.duplicated(subset=['date', 'ticker'], keep=False)
            if duplicate_check.any():
                duplicates = df[duplicate_check]
                issue = f"Found {len(duplicates)} duplicate entries (same date and ticker)"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
        
        # 2. Check for extreme price values
        if 'price' in df.columns:
            # Check for zero or negative prices
            invalid_prices = df[(df['price'] <= 0) & ~pd.isna(df['price'])]
            if len(invalid_prices) > 0:
                issue = f"Found {len(invalid_prices)} rows with zero or negative prices"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
            
            # Check for extreme price outliers (more than 100x median)
            if len(df) > 10:  # Only check if we have enough data
                median_price = df['price'].median()
                if median_price > 0:
                    price_outliers = df[df['price'] > median_price * 100]
                    if len(price_outliers) > 0:
                        issue = f"Found {len(price_outliers)} rows with extreme price values (>100x median)"
                        logger.warning(issue)
                        self.data_quality_issues.append(issue)
        
        # 3. Check for extreme quantity values
        if 'quantity' in df.columns:
            # Check for zero or negative quantities
            invalid_quantities = df[(df['quantity'] <= 0) & ~pd.isna(df['quantity'])]
            if len(invalid_quantities) > 0:
                issue = f"Found {len(invalid_quantities)} rows with zero or negative quantities"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
            
            # Check for extreme quantity outliers
            if len(df) > 10:
                median_quantity = df['quantity'].median()
                if median_quantity > 0:
                    quantity_outliers = df[df['quantity'] > median_quantity * 1000]
                    if len(quantity_outliers) > 0:
                        issue = f"Found {len(quantity_outliers)} rows with extreme quantity values (>1000x median)"
                        logger.warning(issue)
                        self.data_quality_issues.append(issue)
        
        # 4. Check for valid date ranges
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            date_range = dates.max() - dates.min()
            
            if date_range.days < 30:
                issue = f"Portfolio data covers only {date_range.days} days (minimum 30 days recommended)"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
            
            # Check for future dates
            today = pd.Timestamp.now()
            future_dates = dates[dates > today]
            if len(future_dates) > 0:
                issue = f"Found {len(future_dates)} rows with future dates"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
        
        # 5. Check for missing essential columns
        essential_columns = ['date', 'ticker', 'price', 'quantity']
        missing_essential = [col for col in essential_columns if col not in df.columns]
        if missing_essential:
            issue = f"Missing essential columns: {missing_essential}"
            logger.error(issue)
            self.data_quality_issues.append(issue)
        
        # 6. Check for portfolio concentration
        if 'value_krw' in df.columns:
            total_value = df['value_krw'].sum()
            if total_value > 0:
                # Check if portfolio is too small
                if total_value < MIN_PORTFOLIO_VALUE:
                    issue = f"Portfolio value ({total_value:,.0f} KRW) is below minimum threshold ({MIN_PORTFOLIO_VALUE:,.0f} KRW)"
                    logger.warning(issue)
                    self.data_quality_issues.append(issue)
                
                # Check concentration in top holdings
                if len(df) >= 5:
                    df_sorted = df.sort_values('value_krw', ascending=False)
                    top_5_value = df_sorted.head(5)['value_krw'].sum()
                    top_5_pct = (top_5_value / total_value) * 100
                    
                    if top_5_pct > 80:
                        issue = f"High concentration: Top 5 holdings represent {top_5_pct:.1f}% of portfolio"
                        logger.warning(issue)
                        self.data_quality_issues.append(issue)
        
        # 7. Check for data consistency across time periods
        if 'date' in df.columns and 'ticker' in df.columns:
            # Count unique dates and tickers
            unique_dates = df['date'].nunique()
            unique_tickers = df['ticker'].nunique()
            
            if unique_dates < 2:
                issue = f"Only {unique_dates} unique date(s) found - time series analysis limited"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
            
            if unique_tickers < 5:
                issue = f"Only {unique_tickers} unique ticker(s) found - diversification limited"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
        
        # 8. Check for currency consistency
        if 'currency' in df.columns:
            unique_currencies = df['currency'].nunique()
            if unique_currencies > 5:
                issue = f"Portfolio contains {unique_currencies} different currencies - may increase complexity"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
        
        logger.info(f"Data quality validation complete. Found {len(self.data_quality_issues)} issues.")
    
    def _identify_missing_data(self):
        """Identify missing or invalid data in the portfolio."""
        if self.processed_data is None:
            logger.error("No processed data available")
            return
            
        df = self.processed_data
        
        # Identify missing prices
        if 'price' in df.columns:
            missing_prices = df[pd.isna(df['price']) | (df['price'] <= 0)]
            for i in range(len(missing_prices)):
                row = missing_prices.iloc[i]
                self.missing_data["prices"].append({
                    "ticker": str(row['ticker']) if 'ticker' in row else '',
                    "name": str(row['name']) if 'name' in row else '',
                    "date": row['date'] if 'date' in row else None,
                    "currency": str(row['currency']) if 'currency' in row else '',
                    "current_value": None
                })
        
        # Identify missing exchange rates (for non-KRW currencies)
        if 'currency' in df.columns and 'exchange_rate' in df.columns:
            non_krw = df[df['currency'] != 'KRW']
            missing_rates = non_krw[pd.isna(non_krw['exchange_rate']) | (non_krw['exchange_rate'] <= 0)]
            for i in range(len(missing_rates)):
                row = missing_rates.iloc[i]
                self.missing_data["exchange_rates"].append({
                    "currency": str(row['currency']),
                    "date": row['date'] if 'date' in row else None,
                    "current_value": None
                })
        
        # Identify missing sectors
        if 'sector' in df.columns:
            missing_sectors = df[pd.isna(df['sector']) | (df['sector'] == '')]
            for i in range(len(missing_sectors)):
                row = missing_sectors.iloc[i]
                self.missing_data["sectors"].append({
                    "ticker": str(row['ticker']) if 'ticker' in row else '',
                    "name": str(row['name']) if 'name' in row else '',
                    "date": row['date'] if 'date' in row else None,
                    "current_value": None
                })
        
        # Identify missing weights (will be calculated later)
        if 'weight_pct' in df.columns:
            missing_weights = df[pd.isna(df['weight_pct'])]
            for i in range(len(missing_weights)):
                row = missing_weights.iloc[i]
                self.missing_data["weights"].append({
                    "ticker": str(row['ticker']) if 'ticker' in row else '',
                    "date": row['date'] if 'date' in row else None,
                    "current_value": None
                })
        
        # Check for data quality issues
        self._check_data_quality()
        
        logger.info(f"Identified missing data: {len(self.missing_data['prices'])} prices, "
                   f"{len(self.missing_data['exchange_rates'])} exchange rates, "
                   f"{len(self.missing_data['sectors'])} sectors")
    
    def _check_data_quality(self):
        """Check for data quality issues beyond missing data."""
        if self.processed_data is None:
            return
            
        df = self.processed_data
        
        # Check for invalid currencies
        if 'currency' in df.columns:
            invalid_currencies = df[~df['currency'].isin(VALID_CURRENCIES)]
            if len(invalid_currencies) > 0:
                issue = f"Found {len(invalid_currencies)} rows with invalid currencies"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
        
        # Check for invalid sectors
        if 'sector' in df.columns:
            invalid_sectors = df[~df['sector'].isin(VALID_SECTORS)]
            if len(invalid_sectors) > 0:
                issue = f"Found {len(invalid_sectors)} rows with invalid sectors"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
        
        # Check for missing tickers
        if 'ticker' in df.columns:
            missing_tickers = df[pd.isna(df['ticker']) | (df['ticker'] == '')]
            if len(missing_tickers) > 0:
                issue = f"Found {len(missing_tickers)} rows with missing tickers"
                logger.warning(issue)
                self.data_quality_issues.append(issue)
    
    def _calculate_portfolio_summary(self):
        """Calculate basic portfolio summary statistics."""
        if self.processed_data is None:
            return
            
        df = self.processed_data
        
        self.portfolio_summary = {
            "total_rows": len(df),
            "unique_dates": df['date'].nunique() if 'date' in df.columns else 0,
            "unique_tickers": df['ticker'].nunique() if 'ticker' in df.columns else 0,
            "unique_currencies": df['currency'].nunique() if 'currency' in df.columns else 0,
            "unique_sectors": df['sector'].nunique() if 'sector' in df.columns else 0,
            "total_value_krw": df['value_krw'].sum() if 'value_krw' in df.columns else 0,
            "avg_price": df['price'].mean() if 'price' in df.columns else 0,
            "avg_quantity": df['quantity'].mean() if 'quantity' in df.columns else 0,
            "data_quality_issues": len(self.data_quality_issues),
            "missing_data_counts": {
                "prices": len(self.missing_data["prices"]),
                "exchange_rates": len(self.missing_data["exchange_rates"]),
                "sectors": len(self.missing_data["sectors"]),
                "weights": len(self.missing_data["weights"])
            }
        }
        
        logger.info(f"Portfolio summary: {self.portfolio_summary['unique_tickers']} tickers, "
                   f"{self.portfolio_summary['unique_dates']} dates, "
                   f"{self.portfolio_summary['total_value_krw']:,.0f} KRW total value")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        return {
            "summary": self.portfolio_summary,
            "issues": self.data_quality_issues,
            "missing_data": self.missing_data,
            "recommendations": self._generate_data_quality_recommendations()
        }
    
    def _generate_data_quality_recommendations(self) -> List[str]:
        """Generate recommendations based on data quality issues."""
        recommendations = []
        
        if not self.data_quality_issues:
            recommendations.append("Data quality is good. No major issues found.")
        
        # Check for specific issues and provide recommendations
        for issue in self.data_quality_issues:
            if "duplicate entries" in issue:
                recommendations.append("Remove duplicate entries before analysis.")
            
            if "zero or negative prices" in issue:
                recommendations.append("Fix invalid price values before analysis.")
            
            if "zero or negative quantities" in issue:
                recommendations.append("Fix invalid quantity values before analysis.")
            
            if "future dates" in issue:
                recommendations.append("Remove or correct future dates.")
            
            if "below minimum threshold" in issue:
                recommendations.append("Consider adding more holdings to reach minimum portfolio size.")
            
            if "High concentration" in issue:
                recommendations.append("Consider diversifying portfolio to reduce concentration risk.")
            
            if "only" in issue and "unique date" in issue:
                recommendations.append("Add more historical data for better time series analysis.")
            
            if "only" in issue and "unique ticker" in issue:
                recommendations.append("Consider adding more holdings for better diversification.")
        
        # Add general recommendations
        if len(self.missing_data["prices"]) > 0:
            recommendations.append(f"Fetch missing price data for {len(self.missing_data['prices'])} holdings.")
        
        if len(self.missing_data["sectors"]) > 0:
            recommendations.append(f"Add sector information for {len(self.missing_data['sectors'])} holdings.")
        
        return recommendations
    
    def save_updated_data(self, filepath: str) -> bool:
        """Save updated portfolio data to CSV file."""
        try:
            if self.processed_data is None:
                logger.error("No processed data available to save")
                return False
            
            # Create a copy for saving
            save_df = self.processed_data.copy()
            
            # Convert date to string format
            if 'date' in save_df.columns:
                save_df['date'] = save_df['date'].dt.strftime('%Y-%m-%d')
            
            # Save to CSV
            save_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"Saved updated data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving updated data: {e}")
            return False


def test_data_loader():
    """Test function for the PortfolioData class."""
    import sys
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PortfolioData...")
    
    # Create a sample portfolio data
    sample_data = pd.DataFrame({
        '기준일': ['2025-01-01', '2025-01-01', '2025-02-01', '2025-02-01'],
        '종목코드': ['005930', '000660', '005930', '000660'],
        '종목명': ['삼성전자', 'SK하이닉스', '삼성전자', 'SK하이닉스'],
        '수량': [100, 50, 100, 50],
        '현재가': [70000, 150000, 72000, 155000],
        '통화': ['KRW', 'KRW', 'KRW', 'KRW'],
        '섹터': ['Technology', 'Technology', 'Technology', 'Technology']
    })
    
    # Save sample data to CSV
    test_file = "test_portfolio.csv"
    sample_data.to_csv(test_file, index=False, encoding='utf-8-sig')
    
    # Test loading
    portfolio = PortfolioData()
    success = portfolio.load_csv(test_file)
    
    if success:
        print("Data loaded successfully!")
        print(f"Processed data shape: {portfolio.processed_data.shape}")
        print(f"Data quality issues: {len(portfolio.data_quality_issues)}")
        
        # Get quality report
        report = portfolio.get_data_quality_report()
        print(f"\nData Quality Report:")
        print(f"Total rows: {report['summary']['total_rows']}")
        print(f"Unique tickers: {report['summary']['unique_tickers']}")
        print(f"Total value: {report['summary']['total_value_krw']:,.0f} KRW")
        
        if report['issues']:
            print(f"\nIssues found:")
            for issue in report['issues']:
                print(f"  - {issue}")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
    else:
        print("Failed to load data")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\nData loader test complete!")


if __name__ == "__main__":
    test_data_loader()