"""
Configuration settings for Portfolio Factsheet Generator
"""

import os
from datetime import datetime

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# Data output directory
UPDATED_DATA_DIR = os.path.join(OUTPUTS_DIR, "updated_data")

# Ensure directories exist
for directory in [DATA_DIR, TEMPLATES_DIR, REPORTS_DIR, CHARTS_DIR, LOGS_DIR, UPDATED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Cache file paths
SECTOR_CACHE_FILE = os.path.join(DATA_DIR, "sector_cache.json")
FX_CACHE_FILE = os.path.join(DATA_DIR, "fx_cache.json")
PRICE_CACHE_FILE = os.path.join(DATA_DIR, "price_cache.json")

# Portfolio settings
BASE_CURRENCY = "KRW"
BENCHMARKS = ["KOSPI", "S&P", "SPX"]
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate for Sharpe ratio

# Report settings
REPORT_TITLE = "Portfolio Performance Factsheet"
REPORT_AUTHOR = "Portfolio Factsheet Generator"
REPORT_VERSION = "1.0"
DEFAULT_REPORT_FILENAME = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

# API settings
YFINANCE_TIMEOUT = 10  # seconds
EXCHANGERATE_API_URL = "https://api.exchangerate-api.com/v4/latest/{base_currency}"
EXCHANGERATE_TIMEOUT = 5  # seconds
MAX_API_RETRIES = 3
API_RETRY_DELAY = 1  # seconds between retries

# Cache settings
CACHE_ENABLED = True
CACHE_EXPIRY_DAYS = 30  # Days before cache entries expire
MAX_CACHE_SIZE_MB = 50  # Maximum cache size in MB

# Data fetching settings
AUTO_FETCH_ENABLED = True
FETCH_BATCH_SIZE = 5  # Number of items to fetch in parallel
MAX_MANUAL_PROMPTS = 10  # Maximum number of manual prompts before auto-skip

# Ticker mapping for yfinance (adjust suffixes for different markets)
TICKER_MAPPING = {
    ".KQ": ".KS",  # KOSDAQ to Yahoo Finance format
    ".HK": ".HK",  # Hong Kong (same)
    ".SZ": ".SZ",  # Shenzhen (same)
    ".JT": ".T",   # Tokyo to Yahoo Finance format
    ".PA": ".PA",  # Paris (same)
    ".SW": ".SW",  # Swiss (same)
}

# Sector mapping (yfinance sectors to standard GICS sectors)
SECTOR_MAPPING = {
    # yfinance sectors -> Standard GICS sectors
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Industrials": "Industrials",
    "Communication Services": "Communication Services",
    "Utilities": "Utilities",  # Not in current portfolio but for completeness
    "Energy": "Energy",  # Not in current portfolio but for completeness
    "Basic Materials": "Materials",  # Not in current portfolio but for completeness
    "Real Estate": "Real Estate",  # Not in current portfolio but for completeness
}

# Currency symbols for display
CURRENCY_SYMBOLS = {
    "KRW": "₩",
    "USD": "$",
    "EUR": "€",
    "JPY": "¥",
    "CNY": "¥",
    "HKD": "HK$",
    "CHF": "CHF",
}

# Chart settings
CHART_WIDTH = 10  # inches
CHART_HEIGHT = 6  # inches
CHART_DPI = 100  # DPI for saved images
CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Performance calculation settings
ANNUALIZATION_FACTOR = 12  # Monthly data, so sqrt(12) for annualization
MIN_DATA_POINTS_FOR_VOLATILITY = 3  # Minimum months needed to calculate volatility

# GUI settings
GUI_WIDTH = 900
GUI_HEIGHT = 700
GUI_TITLE = "Portfolio Factsheet Generator"
GUI_FONT = ("Arial", 10)
GUI_BG_COLOR = "#f0f0f0"

# Validation settings
MIN_PORTFOLIO_VALUE = 1000  # Minimum portfolio value in KRW to consider valid
MAX_PRICE_CHANGE_PCT = 50  # Maximum allowed price change between months (percentage)
VALID_CURRENCIES = ["KRW", "USD", "EUR", "JPY", "CNY", "HKD", "CHF"]
VALID_SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Financials",
    "Fixed Income",
    "Health Care",
    "Industrials",
    "Information Technology",
]

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOGS_DIR, f"factsheet_{datetime.now().strftime('%Y%m%d')}.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance thresholds for reporting
PERFORMANCE_THRESHOLDS = {
    "excellent_return": 0.15,  # 15%+ annual return
    "good_return": 0.08,       # 8%+ annual return
    "poor_return": 0.00,       # 0% or negative return
    "high_volatility": 0.20,   # 20%+ annual volatility
    "low_volatility": 0.10,    # 10% or less annual volatility
    "good_sharpe": 1.0,        # Sharpe ratio > 1.0
    "excellent_sharpe": 2.0,   # Sharpe ratio > 2.0
}

# Report content settings
INCLUDE_EXECUTIVE_SUMMARY = True
INCLUDE_PERFORMANCE_CHARTS = True
INCLUDE_ALLOCATION_CHARTS = True
INCLUDE_RISK_METRICS = True
INCLUDE_BENCHMARK_COMPARISON = True
INCLUDE_DATA_QUALITY_REPORT = True
INCLUDE_DETAILED_HOLDINGS = True

# Export settings
EXPORT_HTML = True
EXPORT_PDF = False  # Set to True if weasyprint is installed
PDF_DPI = 300
PDF_PAGE_SIZE = "A4"

def get_cache_file_path(cache_type):
    """Get the path for a specific cache file."""
    cache_files = {
        "sector": SECTOR_CACHE_FILE,
        "fx": FX_CACHE_FILE,
        "price": PRICE_CACHE_FILE,
    }
    return cache_files.get(cache_type, SECTOR_CACHE_FILE)

def get_output_path(filename=None, file_type="report"):
    """Get the full path for an output file."""
    if file_type == "report":
        directory = REPORTS_DIR
        if not filename:
            filename = DEFAULT_REPORT_FILENAME
    elif file_type == "chart":
        directory = CHARTS_DIR
        if not filename:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    elif file_type == "log":
        directory = LOGS_DIR
        if not filename:
            filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    else:
        directory = OUTPUTS_DIR
        if not filename:
            filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    return os.path.join(directory, filename)

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check required directories
    for dir_name, dir_path in [
        ("DATA_DIR", DATA_DIR),
        ("TEMPLATES_DIR", TEMPLATES_DIR),
        ("REPORTS_DIR", REPORTS_DIR),
        ("CHARTS_DIR", CHARTS_DIR),
        ("LOGS_DIR", LOGS_DIR),
    ]:
        if not os.path.exists(dir_path):
            errors.append(f"Directory {dir_name} does not exist: {dir_path}")
    
    # Validate settings
    if BASE_CURRENCY not in CURRENCY_SYMBOLS:
        errors.append(f"Base currency {BASE_CURRENCY} not in CURRENCY_SYMBOLS")
    
    if RISK_FREE_RATE < 0 or RISK_FREE_RATE > 1:
        errors.append(f"Risk free rate {RISK_FREE_RATE} should be between 0 and 1")
    
    if YFINANCE_TIMEOUT < 1:
        errors.append(f"YFINANCE_TIMEOUT {YFINANCE_TIMEOUT} should be at least 1 second")
    
    return errors

if __name__ == "__main__":
    # Test configuration
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validated successfully")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Base currency: {BASE_CURRENCY}")
        print(f"Benchmarks: {BENCHMARKS}")