"""
Portfolio calculator module for KRW-based performance calculations.
Calculates returns, risk metrics, allocations, and benchmark comparisons.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any

from config import (
    BASE_CURRENCY, BENCHMARKS, RISK_FREE_RATE,
    ANNUALIZATION_FACTOR, MIN_DATA_POINTS_FOR_VOLATILITY,
    PERFORMANCE_THRESHOLDS
)

# Set up logging
logger = logging.getLogger(__name__)


class PortfolioCalculator:
    """Calculates portfolio performance metrics and analytics."""
    
    def __init__(self, base_currency: str = BASE_CURRENCY):
        self.base_currency = base_currency
        self.portfolio_data = None
        self.results = {}
        
    def calculate_all(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all portfolio metrics.
        
        Args:
            portfolio_data: Processed portfolio data
            
        Returns:
            Dictionary with all calculated metrics
        """
        self.portfolio_data = portfolio_data.copy()
        self.results = {}
        
        try:
            # Calculate basic portfolio values
            self._calculate_portfolio_values()
            
            # Calculate monthly returns
            self._calculate_monthly_returns()
            
            # Calculate cumulative returns
            self._calculate_cumulative_returns()
            
            # Calculate risk metrics
            self._calculate_risk_metrics()
            
            # Calculate allocation metrics
            self._calculate_allocation_metrics()
            
            # Calculate benchmark comparisons
            self._calculate_benchmark_comparisons()
            
            # Calculate performance attribution
            self._calculate_performance_attribution()
            
            # Calculate additional analytics
            self._calculate_additional_analytics()
            
            logger.info("Completed all portfolio calculations")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in portfolio calculations: {e}")
            return {}
    
    def _calculate_portfolio_values(self):
        """Calculate monthly portfolio values in KRW."""
        if self.portfolio_data is None:
            return
            
        df = self.portfolio_data
        
        # Calculate value in KRW for each holding
        df['value_krw'] = df['quantity'] * df['price'] * df['exchange_rate']
        
        # Group by month to get total portfolio value
        monthly_values = df.groupby('year_month')['value_krw'].sum().reset_index()
        monthly_values = monthly_values.sort_values('year_month')
        
        # Store monthly values
        self.results['monthly_values'] = {
            'dates': [str(date) for date in monthly_values['year_month']],
            'values': monthly_values['value_krw'].tolist(),
            'currency': self.base_currency
        }
        
        # Calculate total portfolio statistics
        if len(monthly_values) > 0:
            latest_value = monthly_values.iloc[-1]['value_krw']
            initial_value = monthly_values.iloc[0]['value_krw']
            
            self.results['portfolio_stats'] = {
                'initial_value': initial_value,
                'latest_value': latest_value,
                'total_return_krw': latest_value - initial_value,
                'total_return_pct': ((latest_value / initial_value) - 1) * 100 if initial_value > 0 else 0,
                'num_months': len(monthly_values),
                'avg_monthly_value': monthly_values['value_krw'].mean()
            }
        
        logger.info(f"Calculated portfolio values for {len(monthly_values)} months")
    
    def _calculate_monthly_returns(self):
        """Calculate monthly portfolio returns."""
        if 'monthly_values' not in self.results:
            return
            
        monthly_values = self.results['monthly_values']['values']
        
        if len(monthly_values) < 2:
            logger.warning("Not enough data points for monthly returns calculation")
            return
        
        # Calculate monthly returns
        monthly_returns = []
        for i in range(1, len(monthly_values)):
            if monthly_values[i-1] > 0:
                monthly_return = (monthly_values[i] / monthly_values[i-1]) - 1
                monthly_returns.append(monthly_return)
            else:
                monthly_returns.append(0)
        
        # Calculate statistics
        if monthly_returns:
            monthly_returns_pct = [r * 100 for r in monthly_returns]
            
            self.results['monthly_returns'] = {
                'returns': monthly_returns,
                'returns_pct': monthly_returns_pct,
                'dates': self.results['monthly_values']['dates'][1:],  # Exclude first month
                'avg_monthly_return': np.mean(monthly_returns) * 100,
                'median_monthly_return': np.median(monthly_returns) * 100,
                'best_month': max(monthly_returns) * 100,
                'worst_month': min(monthly_returns) * 100,
                'positive_months': sum(1 for r in monthly_returns if r > 0),
                'negative_months': sum(1 for r in monthly_returns if r < 0),
                'total_months': len(monthly_returns)
            }
            
            logger.info(f"Calculated {len(monthly_returns)} monthly returns")
    
    def _calculate_cumulative_returns(self):
        """Calculate cumulative returns over time."""
        if 'monthly_returns' not in self.results:
            return
            
        monthly_returns = self.results['monthly_returns']['returns']
        
        # Calculate cumulative returns
        cumulative_returns = []
        current_cumulative = 1.0
        
        for monthly_return in monthly_returns:
            current_cumulative *= (1 + monthly_return)
            cumulative_returns.append(current_cumulative - 1)  # Return as excess over 1
        
        # Calculate annualized returns
        num_months = len(monthly_returns)
        if num_months >= 2:
            # Annualized return (geometric) - requires at least 2 months for meaningful calculation
            total_return = cumulative_returns[-1] if cumulative_returns else 0
            annualized_return = (1 + total_return) ** (12 / num_months) - 1
        else:
            annualized_return = None
        
        self.results['cumulative_returns'] = {
            'cumulative': cumulative_returns,
            'cumulative_pct': [r * 100 for r in cumulative_returns],
            'dates': self.results['monthly_returns']['dates'],
            'total_return': cumulative_returns[-1] * 100 if cumulative_returns else 0,
            'annualized_return': annualized_return * 100 if annualized_return else None,
            'num_months': num_months  # Track number of months for report context
        }
    
    def _calculate_risk_metrics(self):
        """Calculate portfolio risk metrics."""
        if 'monthly_returns' not in self.results:
            return
            
        monthly_returns = self.results['monthly_returns']['returns']
        
        if len(monthly_returns) < MIN_DATA_POINTS_FOR_VOLATILITY:
            logger.warning(f"Not enough data points ({len(monthly_returns)}) for risk metrics")
            return
        
        # Calculate volatility (annualized)
        monthly_volatility = np.std(monthly_returns)
        annualized_volatility = monthly_volatility * np.sqrt(ANNUALIZATION_FACTOR)
        
        # Calculate Sharpe ratio (if we have returns)
        avg_monthly_return = np.mean(monthly_returns)
        monthly_risk_free = RISK_FREE_RATE / 12  # Convert annual to monthly
        
        if monthly_volatility > 0:
            monthly_sharpe = (avg_monthly_return - monthly_risk_free) / monthly_volatility
            annualized_sharpe = monthly_sharpe * np.sqrt(ANNUALIZATION_FACTOR)
        else:
            monthly_sharpe = 0
            annualized_sharpe = 0
        
        # Calculate maximum drawdown
        cumulative_returns = [1 + r for r in monthly_returns]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100  # As percentage
        
        # Calculate Sortino ratio (downside risk only)
        downside_returns = [r for r in monthly_returns if r < monthly_risk_free]
        if downside_returns:
            downside_deviation = np.std(downside_returns)
            if downside_deviation > 0:
                sortino_ratio = (avg_monthly_return - monthly_risk_free) / downside_deviation
                sortino_ratio *= np.sqrt(ANNUALIZATION_FACTOR)  # Annualize
            else:
                sortino_ratio = None
        else:
            sortino_ratio = None
        
        self.results['risk_metrics'] = {
            'monthly_volatility': monthly_volatility * 100,
            'annualized_volatility': annualized_volatility * 100,
            'monthly_sharpe': monthly_sharpe,
            'annualized_sharpe': annualized_sharpe,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'downside_deviation': np.std(downside_returns) * 100 if downside_returns else 0,
            'var_95': np.percentile(monthly_returns, 5) * 100,  # 95% VaR
            'cvar_95': np.mean([r for r in monthly_returns if r <= np.percentile(monthly_returns, 5)]) * 100
        }
        
        # Add performance assessment
        self.results['performance_assessment'] = self._assess_performance()
    
    def _assess_performance(self) -> Dict[str, Any]:
        """Assess portfolio performance against thresholds."""
        assessment = {}
        
        if 'risk_metrics' in self.results:
            risk_metrics = self.results['risk_metrics']
            
            # Assess returns
            if 'cumulative_returns' in self.results:
                total_return = self.results['cumulative_returns']['total_return']
                annualized_return = self.results['cumulative_returns'].get('annualized_return')
                
                if annualized_return:
                    if annualized_return >= PERFORMANCE_THRESHOLDS['excellent_return'] * 100:
                        assessment['return_rating'] = 'Excellent'
                    elif annualized_return >= PERFORMANCE_THRESHOLDS['good_return'] * 100:
                        assessment['return_rating'] = 'Good'
                    elif annualized_return >= PERFORMANCE_THRESHOLDS['poor_return'] * 100:
                        assessment['return_rating'] = 'Acceptable'
                    else:
                        assessment['return_rating'] = 'Poor'
                    
                    assessment['annualized_return'] = annualized_return
            
            # Assess volatility
            volatility = risk_metrics['annualized_volatility']
            if volatility <= PERFORMANCE_THRESHOLDS['low_volatility'] * 100:
                assessment['volatility_rating'] = 'Low'
            elif volatility <= PERFORMANCE_THRESHOLDS['high_volatility'] * 100:
                assessment['volatility_rating'] = 'Moderate'
            else:
                assessment['volatility_rating'] = 'High'
            
            # Assess Sharpe ratio
            sharpe = risk_metrics['annualized_sharpe']
            if sharpe >= PERFORMANCE_THRESHOLDS['excellent_sharpe']:
                assessment['sharpe_rating'] = 'Excellent'
            elif sharpe >= PERFORMANCE_THRESHOLDS['good_sharpe']:
                assessment['sharpe_rating'] = 'Good'
            else:
                assessment['sharpe_rating'] = 'Below Target'
            
            assessment['sharpe_ratio'] = sharpe
        
        return assessment
    
    def _calculate_allocation_metrics(self):
        """Calculate sector and geographic allocation metrics."""
        if self.portfolio_data is None:
            return
            
        df = self.portfolio_data
        
        # Get latest month's data
        latest_month = df['year_month'].max()
        latest_data = df[df['year_month'] == latest_month]
        
        # Calculate sector allocation
        sector_allocation = {}
        if 'sector' in latest_data.columns:
            sector_groups = latest_data.groupby('sector')['value_krw'].sum()
            total_value = sector_groups.sum()
            
            if total_value > 0:
                for sector, value in sector_groups.items():
                    if pd.notna(sector):
                        sector_allocation[sector] = {
                            'value': value,
                            'percentage': (value / total_value) * 100,
                            'count': len(latest_data[latest_data['sector'] == sector])
                        }
        
        # Calculate geographic allocation
        geographic_allocation = {}
        if 'country' in latest_data.columns:
            country_groups = latest_data.groupby('country')['value_krw'].sum()
            total_value = country_groups.sum()
            
            if total_value > 0:
                for country, value in country_groups.items():
                    if pd.notna(country):
                        geographic_allocation[country] = {
                            'value': value,
                            'percentage': (value / total_value) * 100,
                            'count': len(latest_data[latest_data['country'] == country])
                        }
        
        # Calculate currency allocation
        currency_allocation = {}
        if 'currency' in latest_data.columns:
            currency_groups = latest_data.groupby('currency')['value_krw'].sum()
            total_value = currency_groups.sum()
            
            if total_value > 0:
                for currency, value in currency_groups.items():
                    if pd.notna(currency):
                        currency_allocation[currency] = {
                            'value': value,
                            'percentage': (value / total_value) * 100,
                            'count': len(latest_data[latest_data['currency'] == currency])
                        }
        
        # Top holdings
        top_holdings = latest_data.nlargest(10, 'value_krw')[['ticker', 'name', 'value_krw', 'weight_pct']]
        top_holdings_list = []
        for _, row in top_holdings.iterrows():
            top_holdings_list.append({
                'ticker': row['ticker'],
                'name': row['name'],
                'value': row['value_krw'],
                'weight': row['weight_pct']
            })
        
        self.results['allocation_metrics'] = {
            'sector_allocation': sector_allocation,
            'geographic_allocation': geographic_allocation,
            'currency_allocation': currency_allocation,
            'top_holdings': top_holdings_list,
            'total_holdings': len(latest_data),
            'latest_month': str(latest_month)
        }
        
        logger.info(f"Calculated allocation metrics for {len(latest_data)} holdings")
    
    def _calculate_benchmark_comparisons(self):
        """Calculate benchmark comparisons (KOSPI, S&P 500)."""
        # Note: In a real implementation, you would fetch benchmark data
        # For now, we'll extract what's available in the portfolio data
        
        if self.portfolio_data is None:
            return
            
        df = self.portfolio_data
        
        benchmark_comparisons = {}
        
        for benchmark in BENCHMARKS:
            # Look for benchmark data in the portfolio
            benchmark_data = df[df['ticker'] == benchmark]
            
            if not benchmark_data.empty:
                # Get monthly values
                monthly_benchmark = benchmark_data.groupby('year_month')['price'].first()
                monthly_benchmark = monthly_benchmark.sort_index()
                
                # Calculate benchmark returns
                if len(monthly_benchmark) > 1:
                    benchmark_returns = monthly_benchmark.pct_change().dropna().tolist()
                    benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod() - 1
                    
                    benchmark_comparisons[benchmark] = {
                        'returns': benchmark_returns,
                        'cumulative_returns': benchmark_cumulative.tolist(),
                        'dates': [str(date) for date in monthly_benchmark.index[1:]],
                        'total_return': benchmark_cumulative.iloc[-1] * 100 if len(benchmark_cumulative) > 0 else 0
                    }
        
        self.results['benchmark_comparisons'] = benchmark_comparisons
        
        # Calculate alpha and beta if we have portfolio returns and benchmark returns
        if ('monthly_returns' in self.results and 
            benchmark_comparisons and 
            'KOSPI' in benchmark_comparisons):
            
            portfolio_returns = self.results['monthly_returns']['returns']
            benchmark_returns = benchmark_comparisons['KOSPI']['returns']
            
            # Align returns (take the minimum length)
            min_len = min(len(portfolio_returns), len(benchmark_returns))
            if min_len > 1:
                port_ret_aligned = portfolio_returns[:min_len]
                bench_ret_aligned = benchmark_returns[:min_len]
                
                # Calculate beta (covariance / variance)
                covariance = np.cov(port_ret_aligned, bench_ret_aligned)[0, 1]
                variance = np.var(bench_ret_aligned)
                
                if variance > 0:
                    beta = covariance / variance
                    
                    # Calculate alpha (average excess return)
                    avg_port_return = np.mean(port_ret_aligned)
                    avg_bench_return = np.mean(bench_ret_aligned)
                    alpha = (avg_port_return - avg_bench_return) * 100  # As percentage
                    
                    self.results['benchmark_analysis'] = {
                        'beta_vs_kospi': beta,
                        'alpha_vs_kospi': alpha,
                        'correlation': np.corrcoef(port_ret_aligned, bench_ret_aligned)[0, 1],
                        'tracking_error': np.std(np.array(port_ret_aligned) - np.array(bench_ret_aligned)) * 100,
                        'information_ratio': alpha / (np.std(np.array(port_ret_aligned) - np.array(bench_ret_aligned)) * 100)
                        if np.std(np.array(port_ret_aligned) - np.array(bench_ret_aligned)) > 0 else None
                    }
    
    def _calculate_performance_attribution(self):
        """Calculate performance attribution by holding."""
        if self.portfolio_data is None:
            return
            
        df = self.portfolio_data
        
        # This is a simplified attribution calculation
        # In a full implementation, you would calculate contribution to returns
        
        # Get unique tickers
        tickers = df['ticker'].unique()
        
        attribution = {}
        for ticker in tickers:
            if ticker in BENCHMARKS:
                continue  # Skip benchmarks
                
            ticker_data = df[df['ticker'] == ticker].sort_values('year_month')
            
            if len(ticker_data) > 1:
                # Calculate ticker return
                initial_value = ticker_data.iloc[0]['value_krw']
                final_value = ticker_data.iloc[-1]['value_krw']
                
                if initial_value > 0:
                    ticker_return = (final_value / initial_value - 1) * 100
                    
                    # Get average weight
                    avg_weight = ticker_data['weight_pct'].mean()
                    
                    attribution[ticker] = {
                        'name': ticker_data.iloc[0]['name'] if 'name' in ticker_data.columns else ticker,
                        'total_return': ticker_return,
                        'avg_weight': avg_weight,
                        'contribution': ticker_return * (avg_weight / 100) if avg_weight > 0 else 0,
                        'sector': ticker_data.iloc[0]['sector'] if 'sector' in ticker_data.columns else 'Unknown'
                    }
        
        # Sort by contribution
        sorted_attribution = sorted(attribution.items(), 
                                   key=lambda x: abs(x[1]['contribution']), 
                                   reverse=True)
        
        self.results['performance_attribution'] = {
            'by_ticker': dict(sorted_attribution[:10]),  # Top 10 contributors
            'total_contributors': len(attribution)
        }
    
    def _calculate_additional_analytics(self):
        """Calculate additional portfolio analytics."""
        if self.portfolio_data is None:
            return
            
        df = self.portfolio_data
        
        # Calculate turnover (simplified)
        # This would normally require transaction data
        turnover_estimate = self._estimate_turnover(df)
        
        # Calculate concentration metrics
        concentration = self._calculate_concentration(df)
        
        # Calculate diversification score
        diversification = self._calculate_diversification_score(df)
        
        self.results['additional_analytics'] = {
            'estimated_turnover': turnover_estimate,
            'concentration_metrics': concentration,
            'diversification_score': diversification,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _estimate_turnover(self, df: pd.DataFrame) -> float:
        """Estimate portfolio turnover (simplified)."""
        try:
            # Count unique tickers per month
            monthly_tickers = df.groupby('year_month')['ticker'].nunique()
            
            if len(monthly_tickers) > 1:
                # Calculate average monthly change in holdings
                monthly_changes = monthly_tickers.diff().abs().mean()
                avg_tickers = monthly_tickers.mean()
                
                if avg_tickers > 0:
                    return (monthly_changes / avg_tickers) * 100
            return 0
        except Exception:
            return 0
    
    def _calculate_concentration(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate concentration metrics."""
        try:
            # Get latest month
            latest_month = df['year_month'].max()
            latest_data = df[df['year_month'] == latest_month]
            
            if len(latest_data) == 0:
                return {}
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            weights = latest_data['weight_pct'] / 100
            hhi = (weights ** 2).sum() * 10000
            
            # Top 5 concentration
            top_5_weight = latest_data.nlargest(5, 'weight_pct')['weight_pct'].sum()
            
            # Top 10 concentration
            top_10_weight = latest_data.nlargest(10, 'weight_pct')['weight_pct'].sum()
            
            return {
                'hhi_index': hhi,
                'top_5_concentration': top_5_weight,
                'top_10_concentration': top_10_weight,
                'num_holdings': len(latest_data)
            }
        except Exception:
            return {}
    
    def _calculate_diversification_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate diversification score (0-100)."""
        try:
            score = 0
            factors = []
            
            # Factor 1: Number of holdings
            latest_month = df['year_month'].max()
            latest_data = df[df['year_month'] == latest_month]
            num_holdings = len(latest_data)
            
            if num_holdings >= 20:
                factors.append(('Holdings Count', 25, 25))
            elif num_holdings >= 10:
                factors.append(('Holdings Count', 20, 25))
            elif num_holdings >= 5:
                factors.append(('Holdings Count', 15, 25))
            else:
                factors.append(('Holdings Count', 10, 25))
            
            # Factor 2: Sector diversification
            if 'sector' in latest_data.columns:
                sectors = latest_data['sector'].nunique()
                if sectors >= 6:
                    factors.append(('Sector Diversification', 25, 25))
                elif sectors >= 4:
                    factors.append(('Sector Diversification', 20, 25))
                elif sectors >= 2:
                    factors.append(('Sector Diversification', 15, 25))
                else:
                    factors.append(('Sector Diversification', 5, 25))
            
            # Factor 3: Geographic diversification
            if 'country' in latest_data.columns:
                countries = latest_data['country'].nunique()
                if countries >= 4:
                    factors.append(('Geographic Diversification', 25, 25))
                elif countries >= 3:
                    factors.append(('Geographic Diversification', 20, 25))
                elif countries >= 2:
                    factors.append(('Geographic Diversification', 15, 25))
                else:
                    factors.append(('Geographic Diversification', 10, 25))
            
            # Factor 4: Currency diversification
            if 'currency' in latest_data.columns:
                currencies = latest_data['currency'].nunique()
                if currencies >= 3:
                    factors.append(('Currency Diversification', 25, 25))
                elif currencies >= 2:
                    factors.append(('Currency Diversification', 20, 25))
                else:
                    factors.append(('Currency Diversification', 15, 25))
            
            # Calculate total score
            total_score = sum(factor[1] for factor in factors)
            max_score = sum(factor[2] for factor in factors)
            
            normalized_score = (total_score / max_score) * 100 if max_score > 0 else 0
            
            return {
                'score': normalized_score,
                'factors': factors,
                'num_holdings': num_holdings
            }
        except Exception:
            return {'score': 0, 'factors': [], 'num_holdings': 0}
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get a summary report of all calculations."""
        summary = {
            'calculation_timestamp': datetime.now().isoformat(),
            'base_currency': self.base_currency,
            'data_points': len(self.portfolio_data) if self.portfolio_data is not None else 0
        }
        
        # Add key metrics from each section
        sections = [
            ('portfolio_stats', 'Portfolio Statistics'),
            ('monthly_returns', 'Monthly Returns'),
            ('cumulative_returns', 'Cumulative Returns'),
            ('risk_metrics', 'Risk Metrics'),
            ('performance_assessment', 'Performance Assessment'),
            ('allocation_metrics', 'Allocation Metrics'),
            ('benchmark_comparisons', 'Benchmark Comparisons'),
            ('performance_attribution', 'Performance Attribution'),
            ('additional_analytics', 'Additional Analytics')
        ]
        
        for key, name in sections:
            if key in self.results:
                summary[name] = self.results[key]
        
        return summary


def test_portfolio_calculator():
    """Test function for the PortfolioCalculator class."""
    import sys
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    dates = pd.date_range('2025-01-01', periods=5, freq='M')
    sample_data = pd.DataFrame({
        'date': dates,
        'year_month': dates.to_period('M'),
        'ticker': ['STOCK1', 'STOCK2', 'STOCK1', 'STOCK2', 'STOCK1'],
        'quantity': [100, 50, 100, 50, 100],
        'price': [100, 200, 110, 190, 115],
        'exchange_rate': [1.0, 1.0, 1.0, 1.0, 1.0],
        'value_krw': [10000, 10000, 11000, 9500, 11500],
        'weight_pct': [50, 50, 53.66, 46.34, 100],
        'sector': ['Technology', 'Healthcare', 'Technology', 'Healthcare', 'Technology'],
        'country': ['US', 'US', 'US', 'US', 'US'],
        'currency': ['USD', 'USD', 'USD', 'USD', 'USD']
    })
    
    print("Testing PortfolioCalculator...")
    
    calculator = PortfolioCalculator()
    results = calculator.calculate_all(sample_data)
    
    if results:
        print("Calculations completed successfully!")
        
        # Print key metrics
        if 'portfolio_stats' in results:
            stats = results['portfolio_stats']
            print(f"\nPortfolio Statistics:")
            print(f"  Initial Value: {stats['initial_value']:,.0f} KRW")
            print(f"  Latest Value: {stats['latest_value']:,.0f} KRW")
            print(f"  Total Return: {stats['total_return_pct']:.2f}%")
        
        if 'monthly_returns' in results:
            returns = results['monthly_returns']
            print(f"\nMonthly Returns:")
            print(f"  Average Monthly Return: {returns['avg_monthly_return']:.2f}%")
            print(f"  Best Month: {returns['best_month']:.2f}%")
            print(f"  Worst Month: {returns['worst_month']:.2f}%")
        
        if 'risk_metrics' in results:
            risk = results['risk_metrics']
            print(f"\nRisk Metrics:")
            print(f"  Annualized Volatility: {risk['annualized_volatility']:.2f}%")
            print(f"  Sharpe Ratio: {risk['annualized_sharpe']:.2f}")
            print(f"  Max Drawdown: {risk['max_drawdown']:.2f}%")
        
        print(f"\nTotal calculation sections: {len(results)}")
        
    else:
        print("Calculations failed!")
    
    print("\nPortfolioCalculator test completed!")


if __name__ == "__main__":
    test_portfolio_calculator()