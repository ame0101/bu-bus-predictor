from flask import Blueprint, render_template, jsonify, request
from app.predictor import BusDelayPredictor
import pandas as pd
import json
import os

main = Blueprint('main', __name__)
predictor = None

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predictions')
def predictions():
    # Load the saved predictions
    predictions_df = pd.read_csv('data/stop_predictions.csv')
    routes = predictions_df['route'].unique()
    return render_template('predictions.html', routes=routes)

@main.route('/api/predictions/<route>')
def get_predictions(route):
    predictions_df = pd.read_csv('data/stop_predictions.csv')
    route_data = predictions_df[predictions_df['route'] == route].to_dict('records')
    return jsonify(route_data)

@main.route('/stats')
def stats():
    stats_df = pd.read_csv('data/stop_delay_statistics.csv')
    return render_template('stats.html', stats=stats_df.to_dict('records'))
