import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smart_home_predictor import SmartHomePredictor

def main():
    # Initialize the predictor
    predictor = SmartHomePredictor()
    
    # Set user inputs
    num_rooms = 5  # 3BR apartment
    budget = 100000  # INR
    priorities = {
        "energy_efficiency": 8,  # High priority on energy efficiency
        "security": 7,          # Good priority on security
        "ease_of_use": 6,       # Medium priority on ease of use
        "scalability": 5        # Lower priority on scalability
    }
    
    # Generate configuration
    print("Generating smart home configuration...")
    configuration = predictor.generate_configuration(num_rooms, budget, priorities)
    
    # Print summary
    print("\nConfiguration Summary:")
    print(f"Total Budget: ₹{configuration['budget']:,.2f}")
    print(f"Total Cost: ₹{configuration['total_cost']:,.2f}")
    print(f"Remaining Budget: ₹{configuration['remaining_budget']:,.2f}")
    
    # Print room allocation summary
    print("\nRoom Allocation:")
    for room in configuration['room_allocations']:
        component_count = len(room['components'])
        print(f"{room['name']}: {component_count} components")
    
    # Visualize component distribution
    print("\nGenerating visualizations...")
    predictor.visualize_component_distribution(configuration)
    plt.savefig("component_distribution.png")
    
    # Visualize cost breakdown
    predictor.visualize_cost_breakdown(configuration)
    plt.savefig("cost_breakdown.png")
    
    # Visualize room allocation
    predictor.visualize_room_allocation(configuration)
    plt.savefig("room_allocation.png")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    predictor.generate_report(configuration, "smart_home_report.html")
    print("Report saved to 'smart_home_report.html'")
    
    print("\nDone! Check the generated files for visualizations and the complete report.")

if __name__ == "__main__":
    main()