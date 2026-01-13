# Portfolio Factsheet Generator

A comprehensive portfolio analysis and reporting tool that generates detailed factsheets from CSV portfolio data. The application provides 3-step GUI for data loading, missing data resolution, and report generation with advanced visualizations.

## Features

### 🎯 **Core Functionality**
- **3-Step GUI Application**: Load CSV → Resolve Missing Data → Generate Report
- **Multi-Currency Support**: Automatic exchange rate fetching (KRW, USD, EUR, JPY, etc.)
- **Data Quality Validation**: Comprehensive input data validation and quality checks
- **Performance Analytics**: Returns, risk metrics, allocation analysis, benchmark comparisons

### 📊 **Advanced Reporting**
- **Interactive HTML Reports**: Professional portfolio factsheets with executive summary
- **6 Visualization Charts**:
  - Cumulative Returns (Line chart)
  - Monthly Returns (Bar chart)
  - Sector Allocation (Pie chart)
  - Geographic Allocation (Horizontal bar chart)
  - Risk-Return Profile (Scatter plot)
  - Portfolio Drawdown (Area chart)
- **PDF Export**: Generate PDF versions of reports
- **Benchmark Comparisons**: KOSPI, S&P 500 (SPX) performance comparison

### 🔧 **Technical Features**
- **Batch Processing**: Optimized API calls for large portfolios
- **Intelligent Caching**: Reduces API calls and improves performance
- **Data Quality Assurance**: 8-step validation process
- **Missing Data Resolution**: Auto-fetch, manual input, or estimation options
- **Korean Encoding Support**: UTF-8, CP949, EUC-KR compatibility

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/[your-username]/portfolio-factsheet-generator.git
cd portfolio-factsheet-generator

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Create `requirements.txt`:
```txt
pandas>=1.5.0
numpy>=1.24.0
yfinance>=0.2.0
requests>=2.28.0
matplotlib>=3.7.0
reportlab>=4.0.0
weasyprint>=60.0  # Optional for PDF generation
```

## Usage

### Quick Start
```bash
cd portfolio_factsheet
python main.py
```

### 3-Step Process
1. **Step 1: Load CSV File**
   - Load portfolio data in Korean CSV format
   - Required columns: `기준일`, `종목코드`, `종목명`, `수량`, `현재가`
   - Optional columns: `통화`, `섹터`, `환율`

2. **Step 2: Resolve Missing Data**
   - **Auto Fetch**: Automatically fetch missing prices, exchange rates, and sectors
   - **Manual Input**: Enter missing values manually
   - **Estimation**: Estimate missing values from available data
   - Save resolved data to new CSV file

3. **Step 3: Generate Report**
   - Calculate portfolio performance metrics
   - Generate HTML report with charts
   - Optionally generate PDF report
   - View report in browser

### Sample CSV Format
```csv
기준일,종목코드,종목명,수량,현재가,통화,섹터
2025-01-01,005930,삼성전자,100,70000,KRW,Technology
2025-01-01,000660,SK하이닉스,50,150000,KRW,Technology
2025-02-01,005930,삼성전자,100,72000,KRW,Technology
```

## Project Structure

```
portfolio_factsheet/
├── main.py                 # Main GUI application
├── config.py              # Configuration and constants
├── modules/
│   ├── data_loader.py     # CSV loading and data validation
│   ├── data_fetcher.py    # API data fetching with caching
│   ├── portfolio_calc.py  # Performance calculations
│   ├── report_gen.py      # HTML/PDF report generation
│   └── sector_mapper.py   # Sector mapping utilities
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## API Integration

### Stock Data (yfinance)
- Automatic price fetching for global stocks
- Sector and industry information
- Historical price data

### Exchange Rates (exchangerate-api.com)
- Real-time and historical exchange rates
- Multiple currency support
- Free tier available

## Data Quality Validation

The application performs comprehensive data quality checks:

1. **Duplicate Detection**: Same date and ticker combinations
2. **Value Validation**: Zero/negative prices and quantities
3. **Date Validation**: Future dates and date ranges
4. **Portfolio Analysis**: Concentration and diversification
5. **Currency Validation**: Supported currencies and consistency
6. **Missing Data**: Identification and resolution options

## Performance Optimization

- **Batch API Calls**: Group requests to minimize API calls
- **Intelligent Caching**: Cache results to avoid重复 requests
- **Memory Optimization**: Efficient data processing for large portfolios
- **Parallel Processing**: Support for concurrent data fetching

## Output Reports

### HTML Report Features
- Executive summary with key metrics
- Performance analysis (returns, volatility, Sharpe ratio)
- Risk analysis (VaR, CVaR, max drawdown)
- Allocation breakdown (sector, geographic, top holdings)
- Benchmark comparisons
- Interactive charts and visualizations

### PDF Report
- Simplified PDF version of the report
- Professional formatting
- Easy sharing and printing

## Development

### Running Tests
```bash
cd portfolio_factsheet
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Type hints for better code clarity
- Comprehensive docstrings

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **yfinance**: For stock price data
- **exchangerate-api.com**: For currency exchange rates
- **matplotlib**: For chart generation
- **reportlab/weasyprint**: For PDF generation

## Support

For issues, questions, or feature requests:
1. Check the [Issues](https://github.com/[your-username]/portfolio-factsheet-generator/issues) page
2. Create a new issue with detailed description

---

**Note**: This tool is for informational purposes only. Past performance is not indicative of future results. Always consult with a financial advisor before making investment decisions.