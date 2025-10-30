# Tesla Model 3 Value Finder & Depreciation Estimator

A Streamlit app that analyzes Tesla Model 3 listings to find the best value cars and estimate cost of ownership with clear visuals and explanations.

## Features

- Clean 4-section layout: Dataset Summary, Depreciation Model, Best Deal Finder, Individual Car Analysis
- Linear regression model (price ~ age + mileage)
- Value metrics: Value Discount (vs predicted), Total/Annual/Monthly Cost, and Monthly Cost incl. opportunity cost (5%)
- Interactive plots (2D, 3D, heatmaps) and an input tool for analyzing a specific car
- Standalone: runs from a CSV file (no cloud dependencies)

## Quickstart

1. Clone this repository
2. Place a CSV named `tesla_long_range_listings.csv` in the repo root
   - Required columns (either raw or cleaned variants):
     - `title`, `subtitle`, `location`, `dealer_rating` (optional)
     - `url` (AutoTrader listing URL; advert id will be extracted)
     - EITHER raw columns: `price`, `mileage`, `registered_year`
     - OR cleaned columns: `price_clean`, `mileage_clean`, `year_clean`, `age_clean`
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
streamlit run tesla_app.py
```

## Data Notes
- The app prefers existing cleaned columns if present; otherwise it derives them from raw fields
- `advert_id` is extracted from `url` (e.g., `/car-details/202501278440582`)
- Default filter shows listings with mileage > 25k (toggle in the sidebar)

## Deployment Tips
- Keep `tesla_long_range_listings.csv` in the repo root so the app runs out-of-the-box
- No secrets or cloud credentials are required
- Works on Streamlit Cloud or any environment with Python 3.9+ and the listed requirements

## License
MIT
