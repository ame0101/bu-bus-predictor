# BU Bus Delay Analysis Results

## Overview
Analysis of delay patterns across Boston University's three shuttle routes reveals significant variations in reliability and performance. Below are detailed findings from our predictive modeling and data analysis.

## Key Findings

### Route Performance Comparison
- **1BU Route**: Most reliable with ~8.1 minutes average delay
- **Comm Ave Route**: Highest delays at ~20.9 minutes average
- **Fenway Route**: Moderate delays at ~13.3 minutes average

### Detailed Analysis by Route

#### 1BU Route (Best Performer)
- Consistent delays across all stops (~8 minutes)
- Key characteristics:
  - Most predictable service pattern
  - Uniform delay distribution
  - Lowest standard deviation in delays
  - High reliability at medical campus stops

#### Comm Ave Route (Most Delayed)
- Average delay of 20.9 minutes
- Notable patterns:
  - Uniform high delays across all stops
  - All stops show similar delay patterns
  - Most affected by traffic conditions
  - Significant delays at major intersections

#### Fenway Route (Moderate Performance)
- Average delay of 13.3 minutes
- Key observations:
  - Consistent delays across all stops
  - Higher delays near Fenway area during events
  - Moderate variability in service reliability
  - Notable impact at Fenway T stop and campus connections

### Stop-Level Analysis

#### Critical Stops
1. **Agganis Arena**
   - Acts as terminal point for all routes
   - Shows consistent delays across routes
   - Key transfer point impact

2. **Central/St Mary's**
   - High traffic intersection impact
   - Consistent delays across all routes
   - Critical connection point

3. **Questrom/Silber Way**
   - Central campus hub
   - Moderate to high delays
   - High passenger volume impact

## Statistical Performance

### Model Performance Metrics
- Mean Absolute Error: 1.23 minutes
- Root Mean Square Error: 4.45 minutes
- R² Score: 0.869

### Delay Distribution
```
Route      Mean    Min    Max
1BU        8.08    0.0    15.0
Comm Ave   20.89   -0.1   39.0
Fenway     13.31   -0.8   24.5
```

## Recommendations

### Immediate Improvements
1. **1BU Route**
   - Maintain current schedule
   - Minor adjustments during peak hours
   - Focus on maintaining consistency

2. **Comm Ave Route**
   - Major schedule revision needed
   - Consider additional buses during peak hours
   - Evaluate stop spacing and locations

3. **Fenway Route**
   - Implement event-based scheduling
   - Adjust timing during game days
   - Review stop placement near Fenway

### Long-term Suggestions
1. **Infrastructure**
   - Dedicated bus lanes where possible
   - Signal priority at major intersections
   - Enhanced bus stop facilities

2. **Operations**
   - Dynamic scheduling system
   - Real-time adjustment capabilities
   - Enhanced driver training

3. **Technology**
   - Improved tracking system
   - Better prediction algorithms
   - Enhanced user interface
# How to Build

## Prerequisites
Python 3.10+
pip (Python package installer)
Git
Make (for Makefile support)
Virtual environment (recommended)

## Step 1: Clone and Setup
### Clone the repository
1. git clone https://github.com/ame0101/bu-bus-predictor.git
2. cd bu-bus-predictor

### Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Step 2: Install Dependencies
### Install all required packages
- make install

### Or manually if Make is not available:
pip install -r requirements.txt

## Step 3: Prepare Data Directories
### Create necessary directories
- mkdir -p data/schedules data/zyte_data app/static/images

### Place your data files:
- Place schedule CSVs in data/schedules/
- Place JSON files in data/zyte_data/

## Step 4: Train Model
### Train the prediction model
- make train

## Or manually:
python scripts/train_model.py

## Step 5: Run Flask Application
### Start the Flask server
- make run

### Or manually:
- flask run

# Common Issues and Solutions

## Package Installation Issues
### If you encounter SSL errors:
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package_name>

### If you need to upgrade pip:
python -m pip install --upgrade pip

## Data Loading Issues
### Ensure your CSV files follow this format:
- stop_name,scheduled_time
- "Agganis Arena","2024-12-10 08:00:00"
- "Target","2024-12-10 08:03:00"

# Ensure your JSON files follow this format:
```
{
    "vehicles": [
        {
            "id": "vehicle_id",
            "position": [lat, long],
            "timestamp": unix_timestamp,
            "route_id": "route_identifier"
        }
    ]
}
```

#### Model Training Issues
# If you encounter memory issues:
export PYTHONMEM=4G  # On Unix
set PYTHONMEM=4G    # On Windows

# Clear existing model:
rm -f bus_delay_model.pkl

### Directory Structure After Build
```
bu-bus-predictor/
├── app/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   ├── templates/
│   └── routes.py
├── data/
│   ├── schedules/
│   │   ├── 1BU_Complete_Timetable.csv
│   │   ├── Comm_Ave_Bus_Timetable.csv
│   │   └── Fenway_Bus_Timetable.csv
│   └── zyte_data/
│       └── [JSON files]
├── scripts/
│   └── train_model.py
├── models/
│   └── bus_delay_model.pkl
├── requirements.txt
├── Makefile
└── README.md
```

## Environment Variables
# Create a .env file with:
FLASK_APP=app
FLASK_ENV=development
PYTHONPATH=${PYTHONPATH}:${PWD}

## Development Setup
#### Install additional development dependencies
pip install -r requirements-dev.txt

### Run tests
make test

### Run linting
make lint

### Clean build files
make clean

### Production Deployment
#### Set production environment
export FLASK_ENV=production

### Run with gunicorn
gunicorn --bind 0.0.0.0:8000 app:app

#### Or with waitress (Windows compatible)
waitress-serve --port=8000 app:app

### Monitoring & Logs
#### View application logs
tail -f logs/app.log

### Monitor system resources
top -p $(pgrep -f 'python')

### Updating
#### Pull latest changes
git pull origin main

### Update dependencies
make update

### Rebuild model
make train



## Methodology Note
All predictions were generated using a Random Forest model trained on:
- Real time GPS data
- Schedule adherence patterns
- Time-of-day variations
- Stop-specific characteristics

## Future Work
1. **Enhanced Prediction**
   - Weather impact integration
   - Special events correlation
   - Passenger load analysis

2. **System Optimization**
   - Route redesign study
   - Stop consolidation analysis
   - Schedule optimization

3. **User Experience**
   - Mobile app development
   - Real-time notifications
   - Crowding predictions

## Contributors
- Amelia Alfonso
- Mario Hysa
- Victoria Lin

## License
MIT License - See [LICENSE](./LICENSE) file for details

---
*Last Updated: December 2024*
