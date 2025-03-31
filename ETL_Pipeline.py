import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def create_mock_data():
    print("Generating synthetic dataset...")
    # Create synthetic data
    np.random.seed(42)
    data = pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C'], size=100),
        'Value1': np.random.randint(10, 100, size=100),
        'Value2': np.random.uniform(1.0, 50.0, size=100),
        'MissingValue': np.random.choice([np.nan, 1, 2], size=100),
    })
    return data

def etl_pipeline(data, output_file):
    # Step 1: Transform the data
    print("Transforming data...")
    # Handle missing values
    data.fillna(method='ffill', inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for column in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Feature scaling
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include='number').columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save the transformed data
    print("Loading data...")
    train_data.to_csv(output_file.replace('.csv', '_train.csv'), index=False)
    test_data.to_csv(output_file.replace('.csv', '_test.csv'), index=False)

    print("ETL process completed successfully!")

# Generate the mock dataset
mock_data = create_mock_data()

# Define output file
output_file = 'mock_output_data.csv'

# Run the ETL pipeline with synthetic data
etl_pipeline(mock_data, output_file)
