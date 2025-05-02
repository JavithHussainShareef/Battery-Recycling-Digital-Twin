import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import random
import time
import sys

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_datasets(bag_path, blackmass_path, metal_prices_path):
    """
    Loads the bag, blackmass, and metal prices datasets from CSV files.
    Exits the program if there's an error in reading any of the files.
    """
    try:
        # Read CSVs for bag, blackmass, and metal prices data
        bag_df = pd.read_csv(bag_path)
        blackmass_df = pd.read_csv(blackmass_path)
        metal_prices_df = pd.read_csv(metal_prices_path, index_col='Date', parse_dates=True)
        
        # Inform the user that the datasets were successfully loaded
        print("[INFO] Datasets loaded successfully.")
        
        return bag_df, blackmass_df, metal_prices_df
    except Exception as e:
        # Print an error message and exit if any dataset fails to load
        print("[ERROR] Unable to load datasets:", e)
        sys.exit()

def merge_and_process_data(bag_df, blackmass_df):
    """
    Merges bag and blackmass data based on the 'Bag Info' and 'Bag ID' fields.
    Converts the 'Process_Date' to datetime, then extracts 'Year' and 'Quarter'.
    Replaces infinite values and drops any rows with missing data.
    """
    try:
        # Merge blackmass_df with selected columns from bag_df
        merged_df = pd.merge(
            blackmass_df,
            bag_df[['Bag ID', 'Source ID', 'Process_Date', 'Weight (KG)']],
            left_on='Bag Info',
            right_on='Bag ID',
            how='left'
        )
        
        # Convert string dates to datetime objects
        merged_df['Process_Date'] = pd.to_datetime(merged_df['Process_Date'])
        
        # Extract the year and quarter from the 'Process_Date'
        merged_df['Year'] = merged_df['Process_Date'].dt.year
        merged_df['Quarter'] = merged_df['Process_Date'].dt.quarter
        
        # Replace infinite values with NaN and drop any rows containing NaN
        merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_df.dropna(inplace=True)
        
        print("[INFO] Data merged and processed successfully.")
        
        return merged_df
    except Exception as e:
        print("[ERROR] Error merging or processing datasets:", e)
        sys.exit()

def normalize_composition(merged_df, elements):
    """
    Normalizes the chemical composition columns using StandardScaler.
    Returns the updated DataFrame, list of normalized column names, and the scaler object.
    """
    # Create new column names (e.g., 'Lithium %', 'Nickel %') for each element
    normalized_cols = [f"{el} %" for el in elements]
    
    # Initialize a standard scaler for normalization
    scaler = StandardScaler()
    
    # Fit and transform the composition columns
    merged_df[normalized_cols] = scaler.fit_transform(merged_df[normalized_cols])
    
    print("[INFO] Chemical composition normalized.")
    return merged_df, normalized_cols, scaler

# Define file paths for input CSVs
bag_dataset_path = "Bag_dataset.csv"
blackmass_path = "Blackmass.csv"
metal_prices_path = "Metal_Prices_2021_2023.csv"

# Load the datasets
bag_df, blackmass_df, metal_prices_df = load_datasets(bag_dataset_path, blackmass_path, metal_prices_path)

# Merge and process the data
merged_df = merge_and_process_data(bag_df, blackmass_df)

# List of elements to be predicted or analyzed
elements = [
    'Titanium', 'Lithium', 'Copper', 'Nickel', 'Cobalt',
    'Manganese', 'Aluminium', 'Magnesium', 'Cadmium', 'Zinc',
    'Lead', 'Chromium', 'Iron'
]

# Normalize chemical composition columns
merged_df, normalized_cols, scaler = normalize_composition(merged_df, elements)

# =============================================================================
# MODEL TRAINING AND SETUP
# =============================================================================

def train_models(merged_df, normalized_cols):
    """
    Splits the merged data into training and test sets.
    Trains a Random Forest and Decision Tree model for predicting the normalized composition.
    Returns the trained models and the train/test splits.
    """
    # Features for training
    X = merged_df[['Year', 'Quarter', 'Source ID', 'Weight (KG)']]
    
    # Target (normalized chemical compositions)
    y = merged_df[normalized_cols]
    
    # Split data into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("[INFO] Random Forest model trained.")

    # Initialize and train the Decision Tree model
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    print("[INFO] Decision Tree model trained.")

    # Inform that we have integrated a hybrid approach
    print("[INFO] Digital Twin approach integrated.")
    print("[INFO] Hybrid model approach integrated (Random Forest & Digital Twin).")

    return rf_model, dt_model, X_train, X_test, y_train, y_test

# Train the models and retrieve the train/test splits
rf_model, dt_model, X_train, X_test, y_train, y_test = train_models(merged_df, normalized_cols)

def train_metal_price_models(metal_prices_df, elements):
    """
    Trains individual Random Forest models for metal price prediction
    based on historical time-series data. Each metal is handled separately.
    """
    metal_price_models = {}
    for metal in elements:
        if metal in metal_prices_df.columns:
            # Extract price values for the metal
            y_prices = metal_prices_df[metal].values
            
            # Train a Random Forest regressor on the timestamp (converted to int64)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(
                metal_prices_df.index.values.reshape(-1, 1).astype('int64') // 10**9,
                y_prices
            )
            # Store the trained model in a dictionary
            metal_price_models[metal] = model

    # Commented out to hide from CLI output:
    # print("[INFO] Metal price prediction models trained.")
    return metal_price_models

# Train the metal price prediction models (commented out print for no CLI output)
metal_price_models = train_metal_price_models(metal_prices_df, elements)

def train_contamination_model(merged_df, normalized_cols):
    """
    Trains an Isolation Forest model to detect contamination in the chemical composition.
    """
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(merged_df[normalized_cols])
    
    # Commented out to hide from CLI output:
    # print("[INFO] Contamination detection model trained.")
    
    return iso_forest

# Train the contamination detection model (commented out print for no CLI output)
iso_forest = train_contamination_model(merged_df, normalized_cols)

# =============================================================================
# DIGITAL TWIN SIMULATION
# =============================================================================

def simulate_digital_twin_effects(composition_df, process_date):
    """
    Simulates the effect of storage time, moisture loss, and efficiency variation
    on the predicted chemical composition. This is a simplified representation
    of how real-world conditions might alter the materials over time.
    """
    simulated = composition_df.copy()
    
    # Calculate how many days have passed since 'process_date'
    today = datetime.today()
    storage_days = (today - process_date).days

    # Degradation factor only applies after 30 days
    degradation_factor = max(0, (storage_days - 30) * 0.0002)
    
    # Random moisture loss between 2% and 10%
    moisture_loss = random.uniform(0.02, 0.1)
    
    # Random efficiency variation factor
    efficiency_variation = random.uniform(0.95, 1.05)

    # Adjust each chemical composition value in the DataFrame
    for col in simulated.columns:
        base_value = simulated[col].values[0]
        adjusted = base_value * (1 - degradation_factor) * (1 - moisture_loss) * efficiency_variation
        simulated[col].values[0] = max(adjusted, 0)
    
    return simulated

# =============================================================================
# PREDICTION FUNCTIONS (HYBRID MODEL)
# =============================================================================

def make_composition_prediction_hybrid(date_str, source_id, weight):
    """
    Generates a hybrid prediction by averaging the outputs of Random Forest and Decision Tree.
    Then applies the Digital Twin simulation to account for real-world effects.
    Returns both the original and simulated compositions.
    """
    try:
        # Convert the user-input date string to a datetime object
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as e:
        print("[ERROR] Invalid date format. Please use YYYY-MM-DD.")
        raise e

    # Extract the year and quarter from the date
    year = date.year
    quarter = (date.month - 1) // 3 + 1

    # Create a single-row DataFrame with the required features
    input_data = pd.DataFrame({
        'Year': [year],
        'Quarter': [quarter],
        'Source ID': [source_id],
        'Weight (KG)': [weight]
    })

    # Predict composition using Random Forest
    rf_pred = rf_model.predict(input_data)
    
    # Predict composition using Decision Tree
    dt_pred = dt_model.predict(input_data)
    
    # Take the average to form the hybrid prediction
    hybrid_pred = (rf_pred + dt_pred) / 2.0

    # Convert normalized predictions back to original scale
    hybrid_original = pd.DataFrame(
        scaler.inverse_transform(hybrid_pred),
        columns=normalized_cols
    )

    # Apply digital twin simulation for real-world adjustment
    hybrid_simulated = simulate_digital_twin_effects(hybrid_original.copy(), date)
    
    return hybrid_original, hybrid_simulated

def detect_contamination(composition_df):
    """
    Uses the Isolation Forest to detect contamination.
    Returns True if contamination is found, False otherwise.
    """
    # Transform the composition_df to normalized space for the model
    norm_input = pd.DataFrame(scaler.transform(composition_df), columns=normalized_cols)
    
    # Isolation Forest returns -1 for outliers (contamination)
    contamination_flag = iso_forest.predict(norm_input)[0]
    
    return contamination_flag == -1

def estimate_batteries(composition_df, weight, battery_type):
    """
    Estimates how many batteries (EV/PHONE/LAPTOP) can be produced
    based on the extracted composition and a predefined material requirement.
    """
    battery_requirements = {
        'EV': {'Lithium %': 8, 'Cobalt %': 5, 'Nickel %': 10},
        'PHONE': {'Lithium %': 3, 'Cobalt %': 2, 'Nickel %': 5},
        'LAPTOP': {'Lithium %': 4, 'Cobalt %': 3, 'Nickel %': 6}
    }
    if battery_type not in battery_requirements:
        print("[ERROR] Invalid battery type. Please enter EV, PHONE, or LAPTOP.")
        return None

    reqs = battery_requirements[battery_type]
    try:
        # Calculate the limiting factor among required metals
        limiting_factor = min(
            (composition_df[metal].values[0] * weight / reqs[metal])
            for metal in reqs
        )
    except Exception as e:
        print("[ERROR] Issue during battery estimation:", e)
        raise e

    return int(limiting_factor)

def predict_metal_price(metal, year):
    """
    Predicts the price of a given metal for a specified year
    using the trained Random Forest metal price model.
    Returns None if the metal model doesn't exist.
    """
    prediction_date = datetime(year, 1, 1).timestamp()
    if metal in metal_price_models:
        return metal_price_models[metal].predict([[int(prediction_date)]])[0]
    return None

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_composition_trends(composition_df):
    """
    Displays a bar chart for the predicted chemical composition of the battery lot.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(composition_df.columns, composition_df.loc[0])
        plt.xticks(rotation=45)
        plt.ylabel("Percentage Composition")
        plt.title("Predicted Chemical Extraction from Lot")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("[ERROR] Issue plotting composition trends:", e)
        raise e

# =============================================================================
# BLOCK: INDIVIDUAL MODEL PREDICTIONS (FOR REFERENCE)
# =============================================================================
# This function demonstrates how to get predictions from each model separately.
# It is not called by default, but can be used for debugging or analysis.

def demo_individual_model_predictions(date_str, source_id, weight):
    """
    Demonstrates predictions from the individual Random Forest and Decision Tree models.
    Useful for debugging or verifying the hybrid approach.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    quarter = (date.month - 1) // 3 + 1

    input_data = pd.DataFrame({
        'Year': [year],
        'Quarter': [quarter],
        'Source ID': [source_id],
        'Weight (KG)': [weight]
    })
    
    # Random Forest prediction
    rf_pred = rf_model.predict(input_data)
    rf_original = pd.DataFrame(
        scaler.inverse_transform(rf_pred),
        columns=normalized_cols
    )
    
    # Decision Tree prediction
    dt_pred = dt_model.predict(input_data)
    dt_original = pd.DataFrame(
        scaler.inverse_transform(dt_pred),
        columns=normalized_cols
    )
    
    print("----- Individual Model Predictions -----")
    print("\n[Random Forest Prediction]:")
    print(rf_original.round(2))
    print("\n[Decision Tree Prediction]:")
    print(dt_original.round(2))
    print("------------------------------------------\n")

# =============================================================================
# CLI FLOW (USING THE HYBRID MODEL)
# =============================================================================

def cli_flow():
    """
    Main command-line interface for the Intelligent Battery Recycling system.
    Interacts with the user, gathers inputs, and performs predictions using the hybrid model.
    """
    print("\n" + "="*50)
    print("🔋 WELCOME TO THE INTELLIGENT BATTERY RECYCLING CLI 🔋")
    print("="*50)
    print("Smarter. Greener. Cleaner.")
    print("Let's turn your e-waste into opportunity!\n")
    time.sleep(1)

    # Prompt user for process date
    while True:
        try:
            date_input = input("Enter process date (YYYY-MM-DD): ").strip()
            datetime.strptime(date_input, "%Y-%m-%d")  # Validate format
            break
        except Exception:
            print("❌ Invalid date format. Please use YYYY-MM-DD.\n")

    # Prompt user for Source ID (1 to 5)
    while True:
        try:
            source_id_input = int(input("Enter your Unique ID (1–5): ").strip())
            if 1 <= source_id_input <= 5:
                break
            else:
                print("❌ Unique ID must be between 1 and 5.\n")
        except Exception:
            print("❌ Please enter a valid integer for Unique ID.\n")

    # Optional location input for traceability
    location = input("Enter location for traceability: ").strip()
    if not location:
        location = "Unknown"

    # Prompt user for weight of the lot in KG
    while True:
        try:
            weight_input = float(input("Enter lot weight in KG (Minimum 5 KG): ").strip())
            if weight_input >= 5:
                break
            else:
                print("❌ Lot size too small. Please enter at least 5 KG.\n")
        except Exception:
            print("❌ Please enter a valid number for lot weight.\n")

    print("\n⏳ Processing chemical extraction prediction using the Hybrid Model...")
    time.sleep(1)

    # Perform the hybrid model prediction
    try:
        original_composition, simulated_composition = make_composition_prediction_hybrid(
            date_input, source_id_input, weight_input
        )
    except Exception as e:
        print("❌ Error during composition prediction:", e)
        sys.exit()

    # Detect contamination using Isolation Forest
    try:
        is_contaminated = detect_contamination(simulated_composition)
    except Exception as e:
        print("❌ Error during contamination detection:", e)
        sys.exit()

    # Display the predicted composition (post-simulation)
    print("\n✅ Predicted Chemical Extraction (Simulated):")
    print(simulated_composition.round(2))

    # Plot the composition in a bar chart
    try:
        plot_composition_trends(simulated_composition)
    except Exception as e:
        print("❌ Error plotting chemical extraction:", e)

    # If contaminated, end the process
    if is_contaminated:
        print("\n⚠️ Contamination detected in this batch. Recycling not recommended.")
        sys.exit()
    else:
        print("\n🟢 Batch is contamination-free and ready for recycling!")

    # Prompt user for battery type (EV, PHONE, or LAPTOP)
    valid_battery_types = {"EV", "PHONE", "LAPTOP"}
    while True:
        battery_type_input = input("\nEnter battery type (EV / Phone / Laptop): ").strip().upper()
        if battery_type_input in valid_battery_types:
            break
        else:
            print("❌ Invalid battery type. Please enter EV, Phone, or Laptop.\n")

    # Estimate how many batteries can be produced
    try:
        batteries_produced = estimate_batteries(
            simulated_composition, weight_input, battery_type_input
        )
    except Exception as e:
        print("❌ Error estimating batteries:", e)
        sys.exit()

    print(f"\n🔁 Estimated {battery_type_input.title()} batteries recycled from this lot: {batteries_produced}")

    # Optional metal price prediction
    interested = input("\nWould you like to predict a metal's price? (yes/no): ").strip().lower()
    if interested == "yes":
        metal_input = input("Enter metal name (e.g., Lithium, Cobalt, Nickel): ").strip().capitalize()
        try:
            predicted_price = predict_metal_price(
                metal_input, datetime.strptime(date_input, "%Y-%m-%d").year
            )
            if predicted_price:
                print(f"\n💰 Predicted price for {metal_input} in {date_input[:4]}: ${predicted_price:.2f} per ton")
            else:
                print("Price data not available for the selected metal.")
        except Exception as e:
            print("❌ Error during metal price prediction:", e)

    print("\n🎉 Process completed! Thank you for using the Smart Recycling CLI.")
    print("Together, we're making the planet cleaner — one battery at a time.\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Uncomment the following line to see individual model predictions for debugging:
    # demo_individual_model_predictions("2025-03-01", 3, 10)

    # Run the CLI flow
    cli_flow()
