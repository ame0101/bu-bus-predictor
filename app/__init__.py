from flask import Flask
from app.routes import main

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    # Register blueprints
    app.register_blueprint(main)
    
    return app