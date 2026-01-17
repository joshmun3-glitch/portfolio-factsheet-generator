"""
Report generator module for creating HTML portfolio factsheet reports.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import base64
from io import BytesIO
import tempfile
from pathlib import Path

from config import (
    PROJECT_ROOT, REPORTS_DIR, CHARTS_DIR,
    REPORT_TITLE, REPORT_AUTHOR, REPORT_VERSION,
    CURRENCY_SYMBOLS, BASE_CURRENCY
)

# Set up logging
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates HTML reports for portfolio factsheets."""
    
    def __init__(self):
        self.template_dir = os.path.join(PROJECT_ROOT, "templates")
        self.reports_dir = REPORTS_DIR
        self.charts_dir = CHARTS_DIR
        
        # Ensure directories exist
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        matplotlib.use('Agg')  # Use non-interactive backend
    
    def generate_report(self, portfolio_data: pd.DataFrame, 
                       calculation_results: Dict[str, Any],
                       title: Optional[str] = None,
                       include_charts: bool = True) -> Optional[str]:
        """
        Generate HTML report for portfolio factsheet.
        
        Args:
            portfolio_data: Processed portfolio data
            calculation_results: Results from portfolio calculator
            title: Report title
            include_charts: Whether to include charts
            
        Returns:
            Path to generated HTML report or None if failed
        """
        try:
            logger.info("Starting report generation...")
            
            # Prepare data for template
            template_data = self._prepare_template_data(
                portfolio_data, calculation_results, title, include_charts
            )
            
            # Generate charts if requested
            if include_charts:
                charts = self._generate_charts(template_data, calculation_results)
                template_data["charts"] = charts
            
            # Generate HTML content
            html_content = self._generate_html_content(template_data)
            
            # Save report
            report_filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report_path = os.path.join(self.reports_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def _prepare_template_data(self, portfolio_data: pd.DataFrame,
                              calculation_results: Dict[str, Any],
                              title: Optional[str] = None,
                              include_charts: bool = True) -> Dict[str, Any]:
        """Prepare data for HTML template."""
        # Basic report info
        report_data = {
            "title": title or REPORT_TITLE,
            "author": REPORT_AUTHOR,
            "version": REPORT_VERSION,
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_currency": BASE_CURRENCY,
            "currency_symbol": CURRENCY_SYMBOLS.get(BASE_CURRENCY, ""),
            "include_charts": include_charts
        }
        
        # Portfolio summary
        if "portfolio_stats" in calculation_results:
            stats = calculation_results["portfolio_stats"]
            report_data["portfolio_stats"] = {
                "initial_value": self._format_currency(stats["initial_value"]),
                "latest_value": self._format_currency(stats["latest_value"]),
                "total_return_krw": self._format_currency(stats["total_return_krw"]),
                "total_return_pct": self._format_percentage(stats["total_return_pct"]),
                "num_months": stats["num_months"],
                "avg_monthly_value": self._format_currency(stats["avg_monthly_value"])
            }
        
        # Monthly returns
        if "monthly_returns" in calculation_results:
            returns = calculation_results["monthly_returns"]
            report_data["monthly_returns"] = {
                "avg_monthly_return": self._format_percentage(returns["avg_monthly_return"]),
                "best_month": self._format_percentage(returns["best_month"]),
                "worst_month": self._format_percentage(returns["worst_month"]),
                "positive_months": returns["positive_months"],
                "negative_months": returns["negative_months"],
                "total_months": returns["total_months"]
            }
            
            # Prepare monthly returns table
            monthly_returns_table = []
            if "dates" in returns and "returns_pct" in returns:
                for i, (date, return_pct) in enumerate(zip(returns["dates"], returns["returns_pct"])):
                    monthly_returns_table.append({
                        "month": date,
                        "return": self._format_percentage(return_pct),
                        "is_positive": return_pct >= 0
                    })
                report_data["monthly_returns_table"] = monthly_returns_table
        
        # Cumulative returns
        if "cumulative_returns" in calculation_results:
            cumulative = calculation_results["cumulative_returns"]
            num_months = cumulative.get("num_months", 0)
            annualized_note = ""
            if num_months > 0 and num_months < 12:
                annualized_note = f" (based on {num_months} months)"

            report_data["cumulative_returns"] = {
                "total_return": self._format_percentage(cumulative["total_return"]),
                "annualized_return": self._format_percentage(cumulative.get("annualized_return", 0))
                    if cumulative.get("annualized_return") else "N/A",
                "annualized_note": annualized_note
            }
        
        # Risk metrics
        if "risk_metrics" in calculation_results:
            risk = calculation_results["risk_metrics"]
            report_data["risk_metrics"] = {
                "annualized_volatility": self._format_percentage(risk["annualized_volatility"]),
                "annualized_sharpe": self._format_number(risk["annualized_sharpe"], 2),
                "max_drawdown": self._format_percentage(risk["max_drawdown"]),
                "sortino_ratio": self._format_number(risk.get("sortino_ratio", 0), 2) 
                    if risk.get("sortino_ratio") else "N/A",
                "var_95": self._format_percentage(risk["var_95"]),
                "cvar_95": self._format_percentage(risk["cvar_95"])
            }
        
        # Performance assessment
        if "performance_assessment" in calculation_results:
            assessment = calculation_results["performance_assessment"]
            report_data["performance_assessment"] = assessment
        
        # Allocation metrics
        if "allocation_metrics" in calculation_results:
            allocation = calculation_results["allocation_metrics"]
            report_data["allocation_metrics"] = allocation
            
            # Prepare sector allocation table
            sector_table = []
            if "sector_allocation" in allocation:
                for sector, data in allocation["sector_allocation"].items():
                    sector_table.append({
                        "sector": sector,
                        "percentage": self._format_percentage(data["percentage"]),
                        "value": self._format_currency(data["value"]),
                        "count": data["count"]
                    })
                # Sort by percentage
                sector_table.sort(key=lambda x: float(x["percentage"].replace('%', '').replace(',', '')), reverse=True)
                report_data["sector_table"] = sector_table
            
            # Prepare geographic allocation table
            geo_table = []
            if "geographic_allocation" in allocation:
                for country, data in allocation["geographic_allocation"].items():
                    geo_table.append({
                        "country": country,
                        "percentage": self._format_percentage(data["percentage"]),
                        "value": self._format_currency(data["value"]),
                        "count": data["count"]
                    })
                # Sort by percentage
                geo_table.sort(key=lambda x: float(x["percentage"].replace('%', '').replace(',', '')), reverse=True)
                report_data["geo_table"] = geo_table
            
            # Prepare top holdings table
            top_holdings_table = []
            if "top_holdings" in allocation:
                for holding in allocation["top_holdings"]:
                    top_holdings_table.append({
                        "ticker": holding["ticker"],
                        "name": holding.get("name", holding["ticker"]),
                        "weight": self._format_percentage(holding["weight"]),
                        "value": self._format_currency(holding["value"])
                    })
                report_data["top_holdings_table"] = top_holdings_table
        
        # Benchmark comparisons
        if "benchmark_comparisons" in calculation_results:
            benchmarks = calculation_results["benchmark_comparisons"]
            benchmark_table = []
            
            for benchmark, data in benchmarks.items():
                benchmark_table.append({
                    "benchmark": benchmark,
                    "total_return": self._format_percentage(data["total_return"])
                })
            
            report_data["benchmark_table"] = benchmark_table
            
            # Benchmark analysis
            if "benchmark_analysis" in calculation_results:
                analysis = calculation_results["benchmark_analysis"]
                report_data["benchmark_analysis"] = {
                    "beta": self._format_number(analysis["beta_vs_kospi"], 2),
                    "alpha": self._format_percentage(analysis["alpha_vs_kospi"]),
                    "correlation": self._format_number(analysis["correlation"], 3),
                    "tracking_error": self._format_percentage(analysis["tracking_error"]),
                    "information_ratio": self._format_number(analysis.get("information_ratio", 0), 2) 
                        if analysis.get("information_ratio") else "N/A"
                }
        
        # 6-month historical performance
        if "monthly_returns" in calculation_results and "cumulative_returns" in calculation_results:
            monthly_data = calculation_results["monthly_returns"]
            cumulative_data = calculation_results["cumulative_returns"]
            
            # Get last 6 months of data (or all if less than 6 months)
            monthly_returns = monthly_data.get("returns", [])
            monthly_dates = monthly_data.get("dates", [])
            cumulative_pct = cumulative_data.get("cumulative_pct", [])
            
            # Create historical performance table (last 6 months)
            historical_table = []
            num_months = min(6, len(monthly_returns))
            
            for i in range(num_months):
                idx = -num_months + i  # Get last N months
                if idx >= 0:  # Ensure we don't go out of bounds
                    historical_table.append({
                        "month": monthly_dates[idx] if idx < len(monthly_dates) else f"Month {i+1}",
                        "monthly_return": self._format_percentage(monthly_returns[idx] * 100),
                        "cumulative_return": self._format_percentage(cumulative_pct[idx] if idx < len(cumulative_pct) else 0)
                    })
            
            report_data["historical_performance"] = historical_table
        
        # Performance attribution
        if "performance_attribution" in calculation_results:
            attribution = calculation_results["performance_attribution"]
            attribution_table = []
            
            if "by_ticker" in attribution:
                for ticker, data in attribution["by_ticker"].items():
                    attribution_table.append({
                        "ticker": ticker,
                        "name": data["name"],
                        "total_return": self._format_percentage(data["total_return"]),
                        "avg_weight": self._format_percentage(data["avg_weight"]),
                        "contribution": self._format_percentage(data["contribution"]),
                        "sector": data["sector"]
                    })
            
            report_data["attribution_table"] = attribution_table
        
        # Additional analytics
        if "additional_analytics" in calculation_results:
            analytics = calculation_results["additional_analytics"]
            report_data["additional_analytics"] = analytics
            
            if "concentration_metrics" in analytics:
                concentration = analytics["concentration_metrics"]
                report_data["concentration_metrics"] = {
                    "hhi_index": self._format_number(concentration.get("hhi_index", 0), 0),
                    "top_5_concentration": self._format_percentage(concentration.get("top_5_concentration", 0)),
                    "top_10_concentration": self._format_percentage(concentration.get("top_10_concentration", 0)),
                    "num_holdings": concentration.get("num_holdings", 0)
                }
            
            if "diversification_score" in analytics:
                diversification = analytics["diversification_score"]
                report_data["diversification_score"] = {
                    "score": self._format_number(diversification.get("score", 0), 1),
                    "num_holdings": diversification.get("num_holdings", 0)
                }
        
        return report_data
    
    def _generate_charts(self, template_data: Dict[str, Any], calculation_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate charts and return base64 encoded images."""
        charts = {}
        
        try:
            # 1. Cumulative Returns Chart
            if "cumulative_returns" in calculation_results and "monthly_returns" in calculation_results:
                cumulative_data = calculation_results["cumulative_returns"]
                monthly_data = calculation_results["monthly_returns"]
                
                if "cumulative_pct" in cumulative_data and "dates" in monthly_data:
                    dates = monthly_data["dates"]
                    cumulative_pct = cumulative_data["cumulative_pct"]
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(dates, cumulative_pct, linewidth=2.5, color='#1a237e')
                    plt.fill_between(dates, cumulative_pct, alpha=0.3, color='#1a237e')
                    plt.title('Cumulative Returns', fontsize=14, fontweight='bold')
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative Return (%)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    charts["cumulative_returns"] = self._fig_to_base64()
                    plt.close()
            
            # 2. Monthly Returns Bar Chart
            if "monthly_returns" in calculation_results:
                monthly_data = calculation_results["monthly_returns"]
                
                if "returns" in monthly_data and "dates" in monthly_data:
                    returns = monthly_data["returns"]
                    dates = monthly_data["dates"]
                    
                    plt.figure(figsize=(12, 6))
                    colors = ['#2e7d32' if r >= 0 else '#c62828' for r in returns]
                    plt.bar(range(len(returns)), returns, color=colors, alpha=0.8)
                    plt.title('Monthly Returns', fontsize=14, fontweight='bold')
                    plt.xlabel('Month')
                    plt.ylabel('Return (%)')
                    plt.xticks(range(len(dates)), [d[:7] for d in dates], rotation=45)
                    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    plt.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    
                    charts["monthly_returns"] = self._fig_to_base64()
                    plt.close()
            
            # 3. Sector Allocation Pie Chart
            if "allocation_metrics" in calculation_results:
                allocation = calculation_results["allocation_metrics"]
                
                if "sector_allocation" in allocation:
                    sector_data = allocation["sector_allocation"]
                    
                    if sector_data:
                        sectors = list(sector_data.keys())
                        percentages = [sector_data[s]["percentage"] for s in sectors]
                        
                        # Create color palette
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                 '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                                 '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
                        colors = colors[:len(sectors)]
                        
                        plt.figure(figsize=(8, 8))
                        plt.pie(
                            percentages, 
                            labels=sectors, 
                            colors=colors,
                            autopct='%1.1f%%',
                            startangle=90,
                            textprops={'fontsize': 9}
                        )
                        plt.title('Sector Allocation', fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        
                        charts["sector_allocation"] = self._fig_to_base64()
                        plt.close()
            
            # 4. Geographic Allocation Chart
            if "allocation_metrics" in calculation_results:
                allocation = calculation_results["allocation_metrics"]
                
                if "geographic_allocation" in allocation:
                    geo_data = allocation["geographic_allocation"]
                    
                    if geo_data:
                        countries = list(geo_data.keys())
                        percentages = [geo_data[c]["percentage"] for c in countries]
                        
                        # Sort by percentage
                        sorted_indices = np.argsort(percentages)[::-1]
                        countries = [countries[i] for i in sorted_indices[:10]]  # Top 10
                        percentages = [percentages[i] for i in sorted_indices[:10]]
                        
                        plt.figure(figsize=(10, 6))
                        # Create blue gradient colors
                        colors = ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5',
                                 '#2196f3', '#1e88e5', '#1976d2', '#1565c0', '#0d47a1']
                        colors = colors[:len(countries)]
                        plt.barh(countries, percentages, color=colors)
                        plt.title('Geographic Allocation (Top 10)', fontsize=14, fontweight='bold')
                        plt.xlabel('Allocation (%)')
                        plt.gca().invert_yaxis()
                        plt.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        
                        charts["geographic_allocation"] = self._fig_to_base64()
                        plt.close()
            
            # 5. Risk-Return Scatter Plot
            if "performance_attribution" in calculation_results:
                attribution = calculation_results["performance_attribution"]
                
                if "by_ticker" in attribution:
                    ticker_data = attribution["by_ticker"]
                    
                    if len(ticker_data) > 1:
                        returns = []
                        volatilities = []
                        tickers = []
                        
                        for ticker, data in ticker_data.items():
                            if "total_return" in data and "volatility" in data:
                                returns.append(data["total_return"])
                                volatilities.append(data["volatility"])
                                tickers.append(ticker)
                        
                        if returns and volatilities:
                            plt.figure(figsize=(10, 6))
                            scatter = plt.scatter(volatilities, returns, alpha=0.7, s=100)
                            plt.title('Risk-Return Profile by Holding', fontsize=14, fontweight='bold')
                            plt.xlabel('Volatility (%)')
                            plt.ylabel('Return (%)')
                            plt.grid(True, alpha=0.3)
                            
                            # Add labels for top holdings
                            for i, ticker in enumerate(tickers[:5]):  # Label top 5
                                plt.annotate(ticker, (volatilities[i], returns[i]), 
                                           fontsize=8, alpha=0.8)
                            
                            plt.tight_layout()
                            charts["risk_return"] = self._fig_to_base64()
                            plt.close()
            
            # 6. Drawdown Chart
            if "risk_metrics" in calculation_results:
                risk_data = calculation_results["risk_metrics"]
                
                if "drawdown_series" in risk_data and "monthly_returns" in calculation_results:
                    drawdown = risk_data["drawdown_series"]
                    monthly_data = calculation_results["monthly_returns"]
                    
                    if "dates" in monthly_data and len(drawdown) == len(monthly_data["dates"]):
                        dates = monthly_data["dates"]
                        
                        plt.figure(figsize=(10, 6))
                        plt.fill_between(range(len(drawdown)), drawdown, alpha=0.5, color='#c62828')
                        plt.plot(drawdown, color='#c62828', linewidth=2)
                        plt.title('Portfolio Drawdown', fontsize=14, fontweight='bold')
                        plt.xlabel('Month')
                        plt.ylabel('Drawdown (%)')
                        plt.xticks(range(len(dates)), [d[:7] for d in dates], rotation=45)
                        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        charts["drawdown"] = self._fig_to_base64()
                        plt.close()
        
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
        
        return charts
    
    def _fig_to_base64(self) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str
    
    def _generate_html_content(self, template_data: Dict[str, Any]) -> str:
        """Generate HTML content from template data."""
        # Simple HTML template
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{template_data['title']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: #1a237e;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        
        .header h1 {{
            margin: 0;
        }}
        
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #1a237e;
            border-bottom: 2px solid #e8eaf6;
            padding-bottom: 10px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #1a237e;
        }}
        
        .metric-card h3 {{
            margin: 0 0 5px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1a237e;
        }}
        
        .metric-positive {{
            color: #2e7d32;
        }}
        
        .metric-negative {{
            color: #c62828;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th {{
            background-color: #e8eaf6;
            color: #1a237e;
            font-weight: 600;
            text-align: left;
            padding: 10px;
            border-bottom: 2px solid #c5cae9;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #eee;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .chart-container {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 3px;
        }}
        
        .chart-title {{
            font-size: 1.1em;
            font-weight: 600;
            color: #1a237e;
            margin-bottom: 10px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{template_data['title']}</h1>
        <p>Generated on {template_data['generation_date']} | Base Currency: {template_data['currency_symbol']}{template_data['base_currency']}</p>
    </div>
    
    <!-- Executive Summary -->
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Return</h3>
                <div class="metric-value {'metric-positive' if 'portfolio_stats' in template_data and float(template_data['portfolio_stats']['total_return_pct'].replace('%', '').replace(',', '')) >= 0 else 'metric-negative'}">
                    {template_data.get('portfolio_stats', {}).get('total_return_pct', 'N/A')}
                </div>
                <p>Over {template_data.get('portfolio_stats', {}).get('num_months', 0)} months</p>
            </div>
            
            <div class="metric-card">
                <h3>Annualized Return</h3>
                <div class="metric-value {'metric-positive' if 'cumulative_returns' in template_data and template_data['cumulative_returns']['annualized_return'] != 'N/A' and float(template_data['cumulative_returns']['annualized_return'].replace('%', '').replace(',', '')) >= 0 else 'metric-negative'}">
                    {template_data.get('cumulative_returns', {}).get('annualized_return', 'N/A')}
                </div>
                <p>Annualized performance{template_data.get('cumulative_returns', {}).get('annualized_note', '')}</p>
            </div>
            
            <div class="metric-card">
                <h3>Annualized Volatility</h3>
                <div class="metric-value">
                    {template_data.get('risk_metrics', {}).get('annualized_volatility', 'N/A')}
                </div>
                <p>Risk measure</p>
            </div>
            
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="metric-value">
                    {template_data.get('risk_metrics', {}).get('annualized_sharpe', 'N/A')}
                </div>
                <p>Risk-adjusted return</p>
            </div>
        </div>
    </div>
    
    <!-- Portfolio Performance -->
    <div class="section">
        <h2>Portfolio Performance</h2>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Initial Portfolio Value</h3>
                <div class="metric-value">
                    {template_data.get('portfolio_stats', {}).get('initial_value', 'N/A')}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Current Portfolio Value</h3>
                <div class="metric-value">
                    {template_data.get('portfolio_stats', {}).get('latest_value', 'N/A')}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Best Month</h3>
                <div class="metric-value metric-positive">
                    {template_data.get('monthly_returns', {}).get('best_month', 'N/A')}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Worst Month</h3>
                <div class="metric-value metric-negative">
                    {template_data.get('monthly_returns', {}).get('worst_month', 'N/A')}
                </div>
            </div>
        </div>
    </div>
    
    <!-- Risk Metrics -->
    <div class="section">
        <h2>Risk Analysis</h2>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Maximum Drawdown</h3>
                <div class="metric-value metric-negative">
                    {template_data.get('risk_metrics', {}).get('max_drawdown', 'N/A')}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>95% Value at Risk</h3>
                <div class="metric-value metric-negative">
                    {template_data.get('risk_metrics', {}).get('var_95', 'N/A')}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Conditional VaR (95%)</h3>
                <div class="metric-value metric-negative">
                    {template_data.get('risk_metrics', {}).get('cvar_95', 'N/A')}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Sortino Ratio</h3>
                <div class="metric-value">
                    {template_data.get('risk_metrics', {}).get('sortino_ratio', 'N/A')}
                </div>
            </div>
        </div>
    </div>
    
    <!-- Portfolio Composition -->
    {self._generate_allocation_html(template_data)}
    
    <!-- Benchmark Comparison -->
    {self._generate_benchmark_html(template_data)}
    
    <!-- Historical Performance -->
    {self._generate_historical_performance_html(template_data)}
    
    <!-- Charts Section -->
    {self._generate_charts_section_html(template_data)}
    
    <div class="footer">
        <p>Generated by {template_data['author']} | {template_data['generation_date']}</p>
        <p>This report is for informational purposes only. Past performance is not indicative of future results.</p>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_charts_section_html(self, template_data: Dict[str, Any]) -> str:
        """Generate HTML for charts section."""
        if "charts" not in template_data or not template_data["charts"]:
            return ""
        
        charts = template_data["charts"]
        html = """
        <div class="section">
            <h2>Visual Analytics</h2>
            <div class="charts-grid">
        """
        
        # Cumulative Returns Chart
        if "cumulative_returns" in charts:
            html += f"""
                <div class="chart-container">
                    <div class="chart-title">Cumulative Returns</div>
                    <img src="data:image/png;base64,{charts['cumulative_returns']}" alt="Cumulative Returns Chart">
                </div>
            """
        
        # Monthly Returns Chart
        if "monthly_returns" in charts:
            html += f"""
                <div class="chart-container">
                    <div class="chart-title">Monthly Returns</div>
                    <img src="data:image/png;base64,{charts['monthly_returns']}" alt="Monthly Returns Chart">
                </div>
            """
        
        # Sector Allocation Chart
        if "sector_allocation" in charts:
            html += f"""
                <div class="chart-container">
                    <div class="chart-title">Sector Allocation</div>
                    <img src="data:image/png;base64,{charts['sector_allocation']}" alt="Sector Allocation Chart">
                </div>
            """
        
        # Geographic Allocation Chart
        if "geographic_allocation" in charts:
            html += f"""
                <div class="chart-container">
                    <div class="chart-title">Geographic Allocation</div>
                    <img src="data:image/png;base64,{charts['geographic_allocation']}" alt="Geographic Allocation Chart">
                </div>
            """
        
        # Risk-Return Chart
        if "risk_return" in charts:
            html += f"""
                <div class="chart-container">
                    <div class="chart-title">Risk-Return Profile</div>
                    <img src="data:image/png;base64,{charts['risk_return']}" alt="Risk-Return Chart">
                </div>
            """
        
        # Drawdown Chart
        if "drawdown" in charts:
            html += f"""
                <div class="chart-container">
                    <div class="chart-title">Portfolio Drawdown</div>
                    <img src="data:image/png;base64,{charts['drawdown']}" alt="Drawdown Chart">
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_allocation_html(self, template_data: Dict[str, Any]) -> str:
        """Generate HTML for allocation sections."""
        html = ""
        
        # Sector Allocation
        if "sector_table" in template_data:
            html += f"""
            <div class="section">
                <h2>Sector Allocation</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Sector</th>
                            <th>Allocation</th>
                            <th>Value</th>
                            <th>Holdings</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for item in template_data["sector_table"]:
                html += f"""
                        <tr>
                            <td>{item['sector']}</td>
                            <td>{item['percentage']}</td>
                            <td>{item['value']}</td>
                            <td>{item['count']}</td>
                        </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        # Top Holdings
        if "top_holdings_table" in template_data:
            html += f"""
            <div class="section">
                <h2>Top Holdings</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Name</th>
                            <th>Weight</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for item in template_data["top_holdings_table"]:
                html += f"""
                        <tr>
                            <td><strong>{item['ticker']}</strong></td>
                            <td>{item['name']}</td>
                            <td>{item['weight']}</td>
                            <td>{item['value']}</td>
                        </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
        
        return html
    
    def _generate_benchmark_html(self, template_data: Dict[str, Any]) -> str:
        """Generate HTML for benchmark comparison."""
        if "benchmark_table" not in template_data:
            return ""
        
        html = """
        <div class="section">
            <h2>Benchmark Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Total Return</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in template_data["benchmark_table"]:
            html += f"""
                    <tr>
                        <td>{item['benchmark']}</td>
                        <td>{item['total_return']}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def _generate_historical_performance_html(self, template_data: Dict[str, Any]) -> str:
        """Generate HTML for 6-month historical performance."""
        if "historical_performance" not in template_data:
            return ""
        
        historical_data = template_data["historical_performance"]
        if not historical_data:
            return ""
        
        html = """
        <div class="section">
            <h2>6-Month Historical Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Monthly Return</th>
                        <th>Cumulative Return</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in historical_data:
            html += f"""
                    <tr>
                        <td>{item['month']}</td>
                        <td>{item['monthly_return']}</td>
                        <td>{item['cumulative_return']}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def _format_currency(self, value: float) -> str:
        """Format currency value."""
        try:
            if value is None or pd.isna(value):
                return "N/A"
            
            value = float(value)
            currency_symbol = CURRENCY_SYMBOLS.get(BASE_CURRENCY, "")
            
            if abs(value) >= 1_000_000_000:
                return f"{currency_symbol}{value/1_000_000_000:,.1f}B"
            elif abs(value) >= 1_000_000:
                return f"{currency_symbol}{value/1_000_000:,.1f}M"
            elif abs(value) >= 1_000:
                return f"{currency_symbol}{value/1_000:,.1f}K"
            else:
                return f"{currency_symbol}{value:,.0f}"
        except:
            return "N/A"
    
    def _format_percentage(self, value: float) -> str:
        """Format percentage value."""
        try:
            if value is None or pd.isna(value):
                return "N/A"
            
            value = float(value)
            return f"{value:+,.2f}%"
        except:
            return "N/A"
    
    def _format_number(self, value: float, decimals: int = 2) -> str:
        """Format number with specified decimals."""
        try:
            if value is None or pd.isna(value):
                return "N/A"
            
            value = float(value)
            return f"{value:,.{decimals}f}"
        except:
            return "N/A"
    
    def generate_pdf_report(self, portfolio_data: pd.DataFrame, 
                          calculation_results: Dict[str, Any],
                          title: Optional[str] = None) -> Optional[str]:
        """
        Generate PDF report for portfolio factsheet.
        
        Args:
            portfolio_data: Processed portfolio data
            calculation_results: Results from portfolio calculator
            title: Report title
            
        Returns:
            Path to generated PDF report or None if failed
        """
        try:
            logger.info("Starting PDF report generation...")
            
            # First generate HTML report
            html_report_path = self.generate_report(
                portfolio_data, calculation_results, title, include_charts=True
            )
            
            if not html_report_path:
                logger.error("Failed to generate HTML report for PDF conversion")
                return None
            
            # Try to convert HTML to PDF using weasyprint if available
            try:
                import weasyprint
                
                # Read HTML content
                with open(html_report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Generate PDF filename
                pdf_filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(self.reports_dir, pdf_filename)
                
                # Convert HTML to PDF
                weasyprint.HTML(string=html_content).write_pdf(pdf_path)
                
                logger.info(f"PDF report generated: {pdf_path}")
                return pdf_path
                
            except ImportError:
                logger.warning("weasyprint not available. Using alternative PDF generation method.")
                return self._generate_pdf_alternative(html_report_path, title)
                
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None
    
    def _generate_pdf_alternative(self, html_report_path: str, title: Optional[str] = None) -> Optional[str]:
        """
        Alternative PDF generation method using reportlab.
        
        Args:
            html_report_path: Path to HTML report
            title: Report title
            
        Returns:
            Path to generated PDF report or None if failed
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            
            # Generate PDF filename
            pdf_filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf_path = os.path.join(self.reports_dir, pdf_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1a237e')
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#1a237e')
            )
            
            normal_style = styles['Normal']
            
            # Build story (content)
            story = []
            
            # Title
            story.append(Paragraph(title or REPORT_TITLE, title_style))
            story.append(Spacer(1, 12))
            
            # Generation info
            story.append(Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                normal_style
            ))
            story.append(Paragraph(
                f"Base Currency: {BASE_CURRENCY}", 
                normal_style
            ))
            story.append(Spacer(1, 24))
            
            # Note about full report
            story.append(Paragraph(
                "Note: This is a simplified PDF version of the portfolio report.", 
                styles['Italic']
            ))
            story.append(Paragraph(
                "For the complete interactive report with charts and detailed analytics,", 
                styles['Italic']
            ))
            story.append(Paragraph(
                f"please refer to the HTML report at: {html_report_path}", 
                styles['Italic']
            ))
            story.append(Spacer(1, 24))
            
            # Footer
            story.append(Paragraph(
                "Generated by Portfolio Factsheet Generator", 
                styles['Italic']
            ))
            story.append(Paragraph(
                "This report is for informational purposes only. Past performance is not indicative of future results.", 
                styles['Italic']
            ))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Alternative PDF report generated: {pdf_path}")
            return pdf_path
            
        except ImportError:
            logger.error("reportlab not available. Cannot generate PDF.")
            return None
        except Exception as e:
            logger.error(f"Error generating alternative PDF: {e}")
            return None


def test_report_generator():
    """Test function for the ReportGenerator class."""
    import sys
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ReportGenerator...")
    
    # Create sample data
    sample_data = {
        "portfolio_stats": {
            "initial_value": 100000000,
            "latest_value": 120000000,
            "total_return_krw": 20000000,
            "total_return_pct": 20.0,
            "num_months": 9,
            "avg_monthly_value": 110000000
        },
        "monthly_returns": {
            "avg_monthly_return": 2.5,
            "best_month": 8.7,
            "worst_month": -3.2,
            "positive_months": 7,
            "negative_months": 2,
            "total_months": 9
        },
        "cumulative_returns": {
            "total_return": 20.0,
            "annualized_return": 26.7
        },
        "risk_metrics": {
            "annualized_volatility": 15.2,
            "annualized_sharpe": 1.75,
            "max_drawdown": -8.5,
            "var_95": -3.2,
            "cvar_95": -5.1
        }
    }
    
    generator = ReportGenerator()
    
    # Create dummy portfolio data
    import pandas as pd
    dummy_data = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=3),
        'ticker': ['A', 'B', 'C'],
        'price': [100, 200, 300]
    })
    
    # Generate report
    report_path = generator.generate_report(
        dummy_data, 
        sample_data,
        title="Test Portfolio Report"
    )
    
    if report_path:
        print(f"Report generated successfully: {report_path}")
        print("ReportGenerator test passed!")
    else:
        print("Failed to generate report")
        print("ReportGenerator test failed!")


if __name__ == "__main__":
    test_report_generator()