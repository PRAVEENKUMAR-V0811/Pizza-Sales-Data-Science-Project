import pandas as pd
from datetime import datetime

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # order_date with correct format (day-month-year)
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)

    return df


def calculate_rfm(df):
    # Reference date is the latest order date + 1
    reference_date = df['order_date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('pizza_category').agg({
        'order_date': lambda x: (reference_date - x.max()).days,
        'order_id': 'count',
        'total_price': 'sum'
    }).reset_index()

    rfm.columns = ['Pizza_Category', 'Recency', 'Frequency', 'Monetary']

    # Fix for identical recency values
    if rfm['Recency'].nunique() > 1:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]).astype(int)
    else:
        rfm['R_Score'] = 1  # or assign a default value

    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    return rfm


def main():
    csv_path = '../pizza_sales.csv'

    df = load_and_prepare_data(csv_path)
    rfm_result = calculate_rfm(df)

    # Save results
    output_path = 'pizza_category_rfm_segment.csv'
    rfm_result.to_csv(output_path, index=False)
    print(f"RFM segmentation saved to: {output_path}")

if __name__ == "__main__":
    main()
