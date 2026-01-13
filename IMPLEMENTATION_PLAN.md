# Portfolio Factsheet Generator - Implementation Plan

## Project Overview
Build a GUI-based portfolio factsheet generator that processes monthly stock portfolio data and generates Fundsmith-style performance reports in HTML/PDF format.

## Core Requirements
1. **Base Currency**: KRW (Korean Won)
2. **Output Formats**: HTML (primary) + PDF (optional)
3. **GUI**: Simple Tkinter interface with manual start button
4. **Missing Data Handling**: yfinance + exchangerate-api → manual input fallback
5. **Benchmarks**: KOSPI & S&P 500 comparison
6. **Sector Mapping**: Auto-detect missing sectors via yfinance
7. **Monthly Reports**: Single report with monthly breakdowns
8. **Caching**: JSON-based cache for API responses
9. **Data Source**: portfolio.csv (Apr-Dec 2025 monthly data)

## Technical Architecture

### File Structure
```
portfolio_factsheet/
├── main.py                    # Tkinter GUI entry point
├── config.py                 # Settings (base currency, APIs, cache settings)
├── data/
│   ├── sector_cache.json     # Cached sector mappings
│   └── fx_cache.json        # Cached exchange rates
├── modules/
│   ├── data_loader.py       # CSV parsing + Korean encoding
│   ├── data_fetcher.py      # yfinance + exchangerate-api integration
│   ├── sector_mapper.py     # Sector detection + GICS mapping
│   ├── portfolio_calc.py    # KRW-based performance calculations
│   ├── report_gen.py        # HTML/Jinja2 report generation
│   ├── chart_gen.py         # Matplotlib static charts
│   └── pdf_exporter.py      # Optional PDF export
├── templates/
│   ├── report_template.html # Fundsmith-style template
│   └── style.css           # Professional styling
└── outputs/
    ├── reports/            # Generated HTML reports
    ├── charts/            # PNG chart images
    └── logs/              # Processing logs
```

## Implementation Phases (7-10 days total)

### Phase 1: Core Engine (Days 1-3)
- CSV parser with Korean encoding support (UTF-8)
- Portfolio data model and validation
- Missing data detection system
- Basic KRW conversion calculations
- Data quality reporting

### Phase 2: API Integration & Caching (Days 4-5)
- yfinance wrapper for stock prices + sectors
- exchangerate-api.com integration for FX rates (free, no key needed)
- Caching system using JSON files
- Error handling and retry logic
- Rate limiting protection

### Phase 3: GUI Development (Days 6-7)
- Tkinter interface with 3-step workflow
- Missing data resolution dialogs
- Progress tracking and status updates
- File selection and report generation buttons
- Simple, intuitive user interface

### Phase 4: Reporting & Charts (Days 8-9)
- HTML template with professional, Fundsmith-inspired styling
- Static charts using matplotlib (performance, allocation, holdings)
- Benchmark comparison tables (KOSPI & S&P 500)
- Data quality report section
- Monthly performance breakdown

### Phase 5: Testing & Polish (Day 10)
- Comprehensive error handling
- User documentation
- Testing with various data scenarios
- Performance optimization
- Final bug fixes

## Key Features

### 1. Smart Data Resolution
- Auto-fetch missing stock prices from yfinance
- Auto-fetch missing FX rates from exchangerate-api.com
- Auto-detect missing sectors via yfinance + GICS mapping
- Manual input fallback with intuitive GUI prompts
- Data validation and sanity checks

### 2. Performance Calculations
- Monthly returns (currency-adjusted to KRW)
- Cumulative performance vs benchmarks
- Sector allocation analysis (9 sectors)
- Geographic allocation analysis (7 countries)
- Top holdings performance attribution
- Risk metrics (volatility, max drawdown, Sharpe ratio)
- Portfolio turnover tracking

### 3. Professional Reporting
- Clean, professional design inspired by Fundsmith
- Monthly performance breakdown tables
- Interactive HTML with embedded static charts
- Optional PDF export capability
- Data quality transparency section
- Benchmark comparison visualizations

### 4. User-Friendly GUI
Simple 3-step process:
```
[Select CSV File] → [Resolve Missing Data] → [Generate Report]
```
- Progress indicators for each step
- Clear error messages with solutions
- One-click report generation
- Preview capability before final generation

## Dependencies

### Required Python Packages:
```txt
pandas>=2.0.0        # Data manipulation
numpy>=1.24.0        # Numerical calculations
matplotlib>=3.7.0    # Chart generation
yfinance>=0.2.0      # Stock price and sector data
requests>=2.31.0     # API calls for exchange rates
jinja2>=3.1.0        # HTML template rendering
tkinter              # GUI (built into Python)
```

### Optional for PDF Export:
```txt
weasyprint>=60.0     # HTML to PDF conversion
```

## Data Sources & APIs

### Primary Sources:
1. **Stock Prices & Sector Data**: yfinance (free, no API key needed)
2. **Exchange Rates**: exchangerate-api.com (free, no API key needed)
3. **Sector Classification**: yfinance + local GICS mapping

### Fallback Strategy:
```
Missing Data → Try yfinance → Try Cache → Manual Input
```

### Sector Mapping Strategy:
- Map yfinance sectors to standard GICS categories
- Cache sector mappings locally
- Handle international ticker formats (.KQ, .HK, .SZ, .JT, .PA, .SW)

## Testing Scenarios

1. **Complete Data Test**: Current portfolio.csv file
2. **Missing December Sectors**: Actual issue in current data
3. **Missing Stock Prices**: Simulate 5-10 missing prices
4. **Missing Exchange Rates**: Remove some FX rates
5. **API Failure Scenarios**: Network issues, rate limits
6. **New Stock Additions**: CFR.SW case (no historical data)
7. **Currency Conversion**: Verify KRW calculations
8. **Benchmark Comparison**: KOSPI and S&P 500 integration

## Risk Mitigation

### Technical Risks:
- **API Rate Limits**: Implement caching + request delays
- **Network Issues**: Retry logic + offline fallback mode
- **Data Quality**: Validation checks + user confirmation
- **Encoding Issues**: UTF-8 with Korean character support
- **Memory Usage**: Process data in chunks for large portfolios

### User Experience Risks:
- **Complex Workflow**: Simplified 3-step process
- **Missing Data Confusion**: Clear prompts and explanations
- **Long Processing Times**: Progress indicators + status updates
- **Error Recovery**: Clear error messages with recovery options

## Configuration Settings

### Key Configurable Parameters:
```python
BASE_CURRENCY = "KRW"
BENCHMARKS = ["KOSPI", "S&P 500"]
REPORT_PERIOD = "Monthly"  # Could be Quarterly, Annual
CACHE_ENABLED = True
CACHE_EXPIRY_DAYS = 30
AUTO_FETCH_ENABLED = True
DEFAULT_RISK_FREE_RATE = 0.02  # 2% for Sharpe ratio
```

## Output Report Structure

### Report Sections:
1. **Executive Summary**: Key performance metrics
2. **Monthly Performance**: Returns vs benchmarks
3. **Portfolio Composition**: Sector and geographic allocation
4. **Top Holdings**: Performance attribution
5. **Risk Analysis**: Volatility, drawdown, correlation
6. **Benchmark Comparison**: Detailed vs KOSPI & S&P 500
7. **Data Quality**: Sources and assumptions used
8. **Appendix**: Detailed monthly data tables

### Chart Types:
1. **Performance Chart**: Portfolio vs benchmarks (line chart)
2. **Sector Allocation**: Current month (pie/donut chart)
3. **Geographic Allocation**: Current month (bar chart)
4. **Top Holdings**: Contribution to returns (horizontal bar)
5. **Monthly Returns**: Heatmap or bar chart
6. **Risk Metrics**: Volatility and drawdown visualization

## Success Criteria

### Functional Requirements:
- Successfully processes portfolio.csv with Korean encoding
- Generates HTML report within 60 seconds
- Handles missing data gracefully with user prompts
- Produces accurate KRW-based performance calculations
- Includes benchmark comparisons
- Creates professional-looking charts

### Non-Functional Requirements:
- Simple, intuitive GUI (3-step process max)
- Clear error messages and recovery options
- Reasonable performance (< 2 minutes for full processing)
- Professional report styling
- Reliable data fetching with fallbacks

## Future Enhancement Ideas

### Phase 2 Features (if needed):
1. **Advanced Risk Metrics**: VaR, CVaR, factor analysis
2. **Peer Comparison**: Compare with other portfolios/funds
3. **Automated Email Reports**: Schedule monthly generation
4. **Web Interface**: Convert to Flask/Django web app
5. **Real-time Data**: Live price updates during market hours
6. **Multiple Portfolio Support**: Compare multiple portfolios
7. **Custom Benchmark Creation**: User-defined benchmarks
8. **Performance Attribution**: Brinson model implementation

## Notes & Assumptions

### Current Data Assumptions:
- portfolio.csv contains Apr-Dec 2025 monthly data
- Korean stocks: .KQ suffix (KOSDAQ)
- International stocks: Various suffixes (.HK, .SZ, .JT, .PA, .SW)
- Benchmarks included: KOSPI and S&P 500
- Base currency for calculations: KRW

### Technical Assumptions:
- Python 3.8+ available on target system
- Internet connection available for API calls
- Sufficient disk space for cache and outputs
- User has basic understanding of portfolio concepts

## Contact & Support
- Primary contact: [Your Name/Team]
- Issue tracking: GitHub Issues (if applicable)
- Documentation: In-code comments + this plan

---
*Last Updated: January 11, 2026*
*Version: 1.0*