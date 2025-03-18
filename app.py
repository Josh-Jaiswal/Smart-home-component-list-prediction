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
    try:
        budget = float(request.form.get('budget', 100000))
    except ValueError:
        budget = 100000
    num_rooms = int(request.form.get('num_rooms', 5))
    
    # Get priorities from sliders (1-10)
    priorities = {
        "energy_efficiency": int(request.form.get('energy_efficiency', 5)),
        "security": int(request.form.get('security', 5)),
        "ease_of_use": int(request.form.get('ease_of_use', 5)),
        "scalability": int(request.form.get('scalability', 5))
    }
    
    # Generate multiple configurations
    configurations = predictor.generate_multiple_configurations(num_rooms, budget, priorities)
    
    # Process configurations for template rendering
    for i, config in enumerate(configurations):
        # Add config_index to each configuration
        config['config_index'] = i
        
        # Add budget and calculate remaining budget
        config['budget'] = budget
        config['remaining_budget'] = budget - config['total_cost']
        
        # Calculate category counts for each configuration
        category_counts = {}
        category_costs = {}
        for comp in config['optimization_result']['selected_components']:
            category = comp['Category']
            price = comp['Price_INR']
            
            # Update category counts
            if category in category_counts:
                category_counts[category] = category_counts[category] + 1
            else:
                category_counts[category] = 1
                
            # Update category costs
            if category in category_costs:
                category_costs[category]['total_cost'] += price
                category_costs[category]['components'].append(comp)
            else:
                category_costs[category] = {
                    'total_cost': price,
                    'components': [comp]
                }
        
        config['category_counts'] = category_counts
        config['category_costs'] = category_costs
        
        # Add index to room allocations for proper tab identification
        for j, room in enumerate(config.get('room_allocations', [])):
            room['index'] = j
            room['id'] = f"{room['name'].replace(' ', '-').lower()}-{i}"
    
    # Save configurations to a temporary JSON file for download purposes
    config_path = os.path.join('static', 'temp_config.json')
    with open(config_path, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json_config = json.dumps(configurations, default=lambda x: float(x) if isinstance(x, np.number) else x)
        f.write(json_config)
    # Add config_index to each configuration
    for i, config in enumerate(configurations):
        config['config_index'] = i
        
        # Add room_index to each room
        for j, room in enumerate(config.get('room_allocations', [])):
            room['room_index'] = j
    # Generate visualizations for each configuration
    all_visualizations = []
    for config in configurations:
        visualizations = {}
        
        # Component distribution visualization as an image
        fig = predictor.visualize_component_distribution(config)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['component_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Cost breakdown visualization as an image (if needed elsewhere)
        # Note: In the merged template, the cost breakdown chart uses Chart.js
        fig = predictor.visualize_cost_breakdown(config)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['cost_breakdown'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Room allocation visualization
        fig = predictor.visualize_room_allocation(config)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['room_allocation'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        all_visualizations.append(visualizations)
    
    # Generate HTML report for the first configuration (default)
    report_html = predictor.generate_report(configurations[0])
    
    # Render the new merged template with the updated interactive elements
    return render_template('results_new.html', 
                           configurations=configurations,
                           all_visualizations=all_visualizations,
                           report_html=report_html,
                           priorities=priorities)

# Route to download a specific report by configuration index
@app.route('/download_report/<int:config_index>')
def download_report(config_index=0):
    # Load the saved configurations
    config_path = os.path.join('static', 'temp_config.json')
    with open(config_path, 'r') as f:
        configurations = json.loads(f.read())
    
    # Get the selected configuration; default to index 0 if out of range
    configuration = configurations[config_index] if config_index < len(configurations) else configurations[0]
    
    # Generate the report HTML file at a static location
    report_path = os.path.join('static', f'smart_home_report_{config_index}.html')
    predictor.generate_report(configuration, report_path)
    
    return redirect(url_for('static', filename=f'smart_home_report_{config_index}.html'))

# Default route for report download (defaults to first configuration)
@app.route('/download_report')
def download_report_default():
    return redirect(url_for('download_report', config_index=0))

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
