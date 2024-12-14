import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pytz
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Eastern Time Zone for Accurate Time Alignment
eastern = pytz.timezone('US/Eastern')

# Constants
MIN_LOOP_TIME = 10    # Minimum loop time in minutes
MAX_LOOP_TIME = 120   # Maximum loop time in minutes (2 hours)
MAX_ALLOWED_DELAY = 60  # Maximum allowed delay in minutes to exclude anomalies
DEBOUNCE_TIME = 5     # Minimum minutes between consecutive stop detections
DISTANCE_THRESHOLD = 200  # Distance threshold in meters to consider bus at stop

# Known route_id mappings
KNOWN_ROUTE_IDS = {
    4014238: "1BU",
    4016220: "Comm Ave",
    4016218: "Fenway"
}

# Route Definitions
route_definitions = {
    "1BU": {
        "stops": [
            {"name": "Agganis Arena", "latitude": 42.353154, "longitude": -71.118131},
            {"name": "Target", "latitude": 42.350753, "longitude": -71.114013},
            {"name": "Central/St Mary's", "latitude": 42.349822, "longitude": -71.106400},
            {"name": "Questrom", "latitude": 42.349025, "longitude": -71.099822},
            {"name": "Kenmore", "latitude": 42.348709, "longitude": -71.095683},
            {"name": "Hynes", "latitude": 42.347074, "longitude": -71.087663},
            {"name": "Mass Ave", "latitude": 42.341708, "longitude": -71.083553},
            {"name": "BU Med Campus", "latitude": 42.335278, "longitude": -71.070821},
            {"name": "Mass Ave (return)", "latitude": 42.341513, "longitude": -71.082889},
            {"name": "Hynes (return)", "latitude": 42.347956, "longitude": -71.087920},
            {"name": "Danielsen", "latitude": 42.350675, "longitude": -71.090231},
            {"name": "610 Beacon (Myles)", "latitude": 42.349611, "longitude": -71.094306},
            {"name": "Questrom @Silber Way", "latitude": 42.349495, "longitude": -71.100807},
            {"name": "Marsh Plaza", "latitude": 42.350142, "longitude": -71.106239},
            {"name": "Buick St corner", "latitude": 42.351344, "longitude": -71.116008},
        ]
    },
    "Comm Ave": {
        "stops": [
            {"name": "Agganis Arena", "latitude": 42.353154, "longitude": -71.118131},
            {"name": "Target", "latitude": 42.350753, "longitude": -71.114013},
            {"name": "Central/St Mary's", "latitude": 42.349822, "longitude": -71.106400},
            {"name": "Questrom", "latitude": 42.349025, "longitude": -71.099822},
            {"name": "Questrom @Silber Way", "latitude": 42.349495, "longitude": -71.100807},
            {"name": "Marsh Plaza", "latitude": 42.350142, "longitude": -71.106239},
            {"name": "Buick St corner", "latitude": 42.351344, "longitude": -71.116008},
        ]
    },
    "Fenway": {
        "stops": [
            {"name": "Agganis Arena", "latitude": 42.353154, "longitude": -71.118131},
            {"name": "Target", "latitude": 42.350753, "longitude": -71.114013},
            {"name": "Central/St Mary's", "latitude": 42.349822, "longitude": -71.106400},
            {"name": "Questrom", "latitude": 42.349025, "longitude": -71.099822},
            {"name": "Fenway T stop", "latitude": 42.345292, "longitude": -71.104552},
            {"name": "Fenway campus", "latitude": 42.342335, "longitude": -71.103816},
            {"name": "New stop", "latitude": 42.3405679, "longitude": -71.1057482},
            {"name": "Return via Fenway T stop", "latitude": 42.345218, "longitude": -71.104243},
            {"name": "Questrom @Silber Way", "latitude": 42.349495, "longitude": -71.100807},
            {"name": "Marsh Plaza", "latitude": 42.350142, "longitude": -71.106239},
            {"name": "Buick St corner", "latitude": 42.351344, "longitude": -71.116008},
        ]
    }
}

class BusDelayPredictor:
    def __init__(self):
        self.model = None
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.feature_columns = [
            'hour', 'minute', 'day_of_week', 'is_peak_hour',
            'speed', 'distance_to_next_stop', 'stop_sequence',
            'route_type'
        ]

    def process_timestamps(self, df):
        """Process timestamps to extract time features"""
        df['datetime'] = pd.to_datetime(df['timestamp_converted'], unit='s')
        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(self.eastern_tz)
        
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        return df

    def prepare_features(self, df):
        """Prepare features for the model"""
        df = self.process_timestamps(df)
        
        df['is_peak_hour'] = df['hour'].apply(
            lambda x: 1 if (x >= 8 and x <= 10) or (x >= 16 and x <= 18) else 0
        )
        
        df['route_type'] = df['route_id'].map(KNOWN_ROUTE_IDS)
        df['route_type'] = pd.Categorical(df['route_type']).codes
        
        df['stop_sequence'] = df.groupby('route_id').cumcount()
        
        if 'distance_to_next_stop' not in df.columns:
            df['distance_to_next_stop'] = 0
            
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[self.feature_columns]

    def train(self, X, y):
        """Train the delay prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        
        return metrics, feature_importance, (y_test, y_pred)

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

def create_output_directories():
    """Create necessary output directories"""
    output_dirs = [
        'outputs',
        'outputs/model',
        'outputs/predictions',
        'outputs/visualizations',
        'outputs/stats'
    ]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    logger.info("Created output directories")

def load_schedule_from_csv(file_path, route_name):
    """Load and validate schedule data from CSV"""
    try:
        df = pd.read_csv(file_path)
        
        if not {'stop_name', 'scheduled_time'}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {file_path}")
        
        df["scheduled_time"] = pd.to_datetime(df["scheduled_time"])
        df["minutes_since_midnight"] = df["scheduled_time"].dt.hour * 60 + df["scheduled_time"].dt.minute
        
        schedule_list = []
        for _, row in df.iterrows():
            schedule_list.append({
                "stop_name": row["stop_name"],
                "scheduled_time": row["minutes_since_midnight"]
            })
        return route_name, schedule_list
    
    except Exception as e:
        logger.error(f"Error processing schedule file {file_path}: {str(e)}")
        return None, None

def process_vehicle_data(file_path):
    """Process vehicle data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        vehicles = []
        for entry in data:
            if 'vehicles' in entry:
                for vehicle in entry['vehicles']:
                    if isinstance(vehicle.get('position'), list):
                        vehicle['timestamp_converted'] = vehicle['timestamp'] // 1000
                        vehicles.append(vehicle)
        
        return vehicles
    except Exception as e:
        logger.error(f"Error processing vehicle file {file_path}: {str(e)}")
        return None

def calculate_delay(df, schedules):
    """Calculate delays for all vehicles"""
    df['route_name'] = df['route_id'].map(KNOWN_ROUTE_IDS)
    df['datetime'] = pd.to_datetime(df['timestamp_converted'], unit='s')
    df['minutes_since_midnight'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    
    delays = []
    for _, row in df.iterrows():
        route_schedule = schedules.get(row['route_name'], [])
        if route_schedule:
            scheduled_times = [s['scheduled_time'] for s in route_schedule]
            actual_time = row['minutes_since_midnight']
            
            closest_time = min(scheduled_times, key=lambda x: abs(x - actual_time))
            delay = actual_time - closest_time
            
            if delay > 720:  # More than 12 hours
                delay -= 1440  # Subtract 24 hours
            elif delay < -720:
                delay += 1440
                
            if abs(delay) <= MAX_ALLOWED_DELAY:
                delays.append(delay)
            else:
                delays.append(None)
        else:
            delays.append(None)
    
    return delays

def generate_predictions(predictor, df, route_definitions, time_ranges=None):
    """Generate predictions with proper handling of duplicate stops"""
    if time_ranges is None:
        time_ranges = [(h, 0) for h in range(7, 23)]
    
    predictions = []
    
    for route_name, route_info in route_definitions.items():
        print(f"\nGenerating predictions for {route_name}...")
        
        unique_stops = []
        for stop in route_info['stops']:
            if stop['name'] not in unique_stops:
                unique_stops.append(stop['name'])
        
        for stop_name in unique_stops:
            print(f"  Processing stop: {stop_name}")
            
            for hour, minute in time_ranges:
                test_data = pd.DataFrame({
                    'hour': [hour],
                    'minute': [minute],
                    'day_of_week': [0],
                    'is_peak_hour': [1 if (hour >= 8 and hour <= 10) or (hour >= 16 and hour <= 18) else 0],
                    'speed': [df['speed'].mean()],
                    'distance_to_next_stop': [0],
                    'stop_sequence': [unique_stops.index(stop_name)],
                    'route_type': [list(KNOWN_ROUTE_IDS.values()).index(route_name)]
                })
                
                predicted_delay = predictor.predict(test_data)[0]
                
                predictions.append({
                    'route': route_name,
                    'stop': stop_name,
                    'time': f"{hour:02d}:{minute:02d}",
                    'predicted_delay': round(predicted_delay, 1)
                })
    
    return pd.DataFrame(predictions)

def create_visualizations(predictions_df, route_definitions):
    """Create all visualizations"""
    
    # Set style for better-looking graphs
    plt.style.use('seaborn')
    
    for route_name in predictions_df['route'].unique():
        route_data = predictions_df[predictions_df['route'] == route_name]
        
        # Average delays by stop
        plt.figure(figsize=(15, 6))
        avg_delays = route_data.groupby('stop')['predicted_delay'].mean()
        plt.bar(range(len(avg_delays)), avg_delays.values)
        plt.xticks(range(len(avg_delays)), avg_delays.index, rotation=45, ha='right')
        plt.title(f'Average Predicted Delays by Stop - {route_name}')
        plt.xlabel('Stop Name')
        plt.ylabel('Average Delay (minutes)')
        plt.tight_layout()
        plt.savefig(f'outputs/visualizations/delays_by_stop_{route_name.replace(" ", "_")}.png')
        plt.close()
    
    # Route comparison
    plt.figure(figsize=(10, 6))
    route_avg_delays = predictions_df.groupby('route')['predicted_delay'].mean()
    plt.bar(route_avg_delays.index, route_avg_delays.values)
    plt.title('Average Delays by Route')
    plt.xlabel('Route')
    plt.ylabel('Average Delay (minutes)')
    plt.tight_layout()
    plt.savefig('outputs/visualizations/route_comparison.png')
    plt.close()

def create_summary_stats(predictions_df):
    """Create summary statistics"""
    summary_stats = []
    
    for route in predictions_df['route'].unique():
        route_data = predictions_df[predictions_df['route'] == route]
        for stop in route_data['stop'].unique():
            stop_data = route_data[route_data['stop'] == stop]
            stats = {
                'route': route,
                'stop': stop,
                'avg_delay': stop_data['predicted_delay'].mean(),
                'max_delay': stop_data['predicted_delay'].max(),
                'min_delay': stop_data['predicted_delay'].min(),
                'std_delay': stop_data['predicted_delay'].std(),
                'peak_hour_delay': stop_data[stop_data['time'].isin(['08:00', '09:00', '17:00', '18:00'])]['predicted_delay'].mean(),
                'off_peak_delay': stop_data[~stop_data['time'].isin(['08:00', '09:00', '17:00', '18:00'])]['predicted_delay'].mean()
            }
            summary_stats.append(stats)
    
    stats_df = pd.DataFrame(summary_stats)
    stats_df.to_csv('outputs/stats/stop_delay_statistics.csv', index=False)
    return stats_df

def main():
    # Create output directories
    create_output_directories()
    
    # Initialize predictor
    predictor = BusDelayPredictor()
    
    # Load schedules
    print("Loading schedules...")
    schedules = {}
    schedule_files = [
        ("1BU_Complete_Timetable.csv", "1BU"),
        ("Comm_Ave_Bus_Timetable.csv", "Comm Ave"),
        ("Fenway_Bus_Timetable.csv", "Fenway")
    ]
    
    for file_name, route_name in schedule_files:
        route_name, schedule = load_schedule_from_csv(f"data/schedules/{file_name}", route_name)
        if route_name and schedule:
            schedules[route_name] = schedule
            print(f"Loaded schedule for {route_name}")
    
    # Process vehicle data
    print("Processing vehicle data...")
    all_vehicle_data = []
    data_folder = "data/zyte_data"
    
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".json"):
            vehicles = process_vehicle_data(os.path.join(data_folder, file_name))
            if vehicles:
                all_vehicle_data.extend(vehicles)
    
    # Create DataFrame from processed data
    df = pd.DataFrame(all_vehicle_data)
    
    if df.empty:
        print("No data was loaded. Please check your input files.")
        return
        
    print(f"Loaded {len(df)} vehicle records")
    
    # Calculate delays
    print("Calculating delays...")
    df['delay'] = calculate_delay(df, schedules)
    
    # Remove rows with None delays
    df = df.dropna(subset=['delay'])
    
    if df.empty:
        print("No valid delay calculations. Please check your schedule and vehicle data.")
        return
        
    print(f"Processed {len(df)} records with valid delays")
    
    # Prepare features and train model
    print("Training model...")
    features = predictor.prepare_features(df)
    metrics, importance, (y_test, y_pred) = predictor.train(features, df['delay'])
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error: {metrics['mae']:.2f} minutes")
    print(f"Root Mean Square Error: {metrics['rmse']:.2f} minutes")
    print(f"R² Score: {metrics['r2']:.3f}")
    
    # Save feature importance
    importance.to_csv('outputs/stats/feature_importance.csv', index=False)
    print("Saved feature importance to outputs/stats/feature_importance.csv")
    
    # Generate predictions
    print("\nGenerating predictions for all stops...")
    predictions_df = generate_predictions(predictor, df, route_definitions)
    
    # Save predictions
    predictions_df.to_csv('outputs/predictions/stop_predictions.csv', index=False)
    print("Saved predictions to outputs/predictions/stop_predictions.csv")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(predictions_df, route_definitions)
    
    # Create summary statistics
    stats_df = create_summary_stats(predictions_df)
    print("Saved summary statistics to outputs/stats/stop_delay_statistics.csv")
    
    # Save model
    with open('outputs/model/bus_delay_model.pkl', 'wb') as f:
        pickle.dump(predictor.model, f)
    print("Saved trained model to outputs/model/bus_delay_model.pkl")
    
    print("\nProcessing complete! Check the 'outputs' directory for all results.")

if __name__ == "__main__":
    main()