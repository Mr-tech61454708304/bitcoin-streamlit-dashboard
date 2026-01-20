# Institutional Bitcoin Market Analytics Platform

A professional, production-grade Streamlit dashboard for comprehensive Bitcoin market analysis.

## Features

- **Robust Data Engineering**: Schema validation, datetime indexing, missing value handling, and data integrity checks
- **Deep Exploratory Data Analysis**: Trend analysis, seasonality, distribution statistics (skewness/kurtosis), drawdown analysis
- **Financial KPIs**: Current price, multi-period changes, ATH/ATL with dates, annualized volatility, risk metrics
- **Advanced Time-Series Analytics**: Multiple moving averages, trend smoothing, price-volume correlation
- **Interactive Visualizations**: Publication-grade charts with technical indicators and overlays
- **Dynamic Controls**: Date range selection, visualization toggles, responsive updates
- **Automated Insights**: Analytical, data-driven textual summaries
- **Professional UI**: Metric cards, expandable sections, clean layout

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bitcoin.git
   cd bitcoin
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to access the dashboard.

## Data

The dashboard uses `bitcoin.csv` containing historical Bitcoin market data with columns: Date, Close, High, Low, Open, Volume.

## Architecture

- `app.py`: Main Streamlit application with modular functions
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore rules
- `bitcoin.csv`: Historical market data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License