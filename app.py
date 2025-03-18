from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from smart_home_predictor import SmartHomePredictor

app = Flask(__name__)

# Initialize the predictor
predictor = SmartHomePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get user inputs from form
    budget = float(request.form.get('budget', 100000))
    num_rooms = int(request.form.get('num_rooms', 5))
    
    # Get priorities from sliders (1-10)
    priorities = {
        "energy_efficiency": int(request.form.get('energy_efficiency', 5)),
        "security": int(request.form.get('security', 5)),
        "ease_of_use": int(request.form.get('ease_of_use', 5)),
        "scalability": int(request.form.get('scalability', 5))
    }
    
    # Generate configuration
    configuration = predictor.generate_configuration(num_rooms, budget, priorities)
    
    # Save configuration to session or temporary file
    config_path = os.path.join('static', 'temp_config.json')
    with open(config_path, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json_config = json.dumps(configuration, default=lambda x: float(x) if isinstance(x, np.number) else x)
        f.write(json_config)
    
    # Generate visualizations and save as base64 strings
    visualizations = {}
    
    # Component distribution
    fig = predictor.visualize_component_distribution(configuration)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visualizations['component_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Cost breakdown
    fig = predictor.visualize_cost_breakdown(configuration)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visualizations['cost_breakdown'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Room allocation
    fig = predictor.visualize_room_allocation(configuration)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visualizations['room_allocation'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Generate HTML report
    report_html = predictor.generate_report(configuration)
    
    return render_template('results.html', 
                           configuration=configuration,
                           visualizations=visualizations,
                           report_html=report_html,
                           priorities=priorities)

@app.route('/download_report')
def download_report():
    # Load the saved configuration
    config_path = os.path.join('static', 'temp_config.json')
    with open(config_path, 'r') as f:
        config_json = f.read()
        configuration = json.loads(config_json)
    
    # Generate HTML report
    report_path = os.path.join('static', 'smart_home_report.html')
    predictor.generate_report(configuration, report_path)
    
    return redirect(url_for('static', filename='smart_home_report.html'))

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)