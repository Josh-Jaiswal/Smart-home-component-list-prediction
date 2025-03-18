import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
import random
from collections import defaultdict

class SmartHomePredictor:
    def __init__(self, data_path="synthetic_smart_home_components.csv"):
        """
        Initialize the Smart Home Predictor with the dataset
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing smart home component data
        """
        # Load and preprocess the data
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
        # Initialize room types and their typical components
        self.room_types = {
            "Living Room": {
                "Lighting": 3,  # e.g., 3 light sources
                "Security": 1,  # e.g., 1 camera
                "HVAC": 1,      # e.g., 1 thermostat
                "Energy Management": 2  # e.g., 2 smart plugs
            },
            "Bedroom": {
                "Lighting": 2,
                "Security": 0,
                "HVAC": 1,
                "Energy Management": 1
            },
            "Kitchen": {
                "Lighting": 2,
                "Security": 0,
                "HVAC": 0,
                "Energy Management": 3
            },
            "Bathroom": {
                "Lighting": 1,
                "Security": 0,
                "HVAC": 1,
                "Energy Management": 0
            },
            "Hallway": {
                "Lighting": 1,
                "Security": 1,
                "HVAC": 0,
                "Energy Management": 0
            },
            "Entrance": {
                "Lighting": 1,
                "Security": 2,  # e.g., doorbell camera and smart lock
                "HVAC": 0,
                "Energy Management": 0
            }
        }
        
        # Default weights for composite score calculation
        self.default_weights = {
            "efficiency": 0.4,
            "reliability": 0.4,
            "price": 0.2
        }
    
    def preprocess_data(self):
        """
        Preprocess the data: check for missing values, duplicates, and normalize numerical features
        """
        # Check for missing values
        if self.df.isnull().sum().sum() > 0:
            print(f"Warning: Found {self.df.isnull().sum().sum()} missing values")
            self.df = self.df.dropna()
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Warning: Found {duplicates} duplicate entries")
            self.df = self.df.drop_duplicates()
        
        # Normalize numerical features
        for category in self.df['Category'].unique():
            category_df = self.df[self.df['Category'] == category]
            
            # Normalize price within each category
            min_price = category_df['Price_INR'].min()
            max_price = category_df['Price_INR'].max()
            
            # Create normalized price column (0-1 scale, where 0 is cheapest)
            self.df.loc[self.df['Category'] == category, 'Normalized_Price'] = \
                (self.df.loc[self.df['Category'] == category, 'Price_INR'] - min_price) / (max_price - min_price)
    
    def calculate_composite_score(self, weights=None):
        """
        Calculate composite score for each component based on weights
        
        Parameters:
        -----------
        weights : dict
            Dictionary with weights for efficiency, reliability, and price
        
        Returns:
        --------
        DataFrame with added composite score column
        """
        if weights is None:
            weights = self.default_weights
        
        # Ensure weights are normalized
        total = sum(weights.values())
        normalized_weights = {k: v/total for k, v in weights.items()}
        
        # Calculate composite score
        self.df['Composite_Score'] = (
            normalized_weights['efficiency'] * self.df['Efficiency'] +
            normalized_weights['reliability'] * self.df['Reliability'] -
            normalized_weights['price'] * self.df['Normalized_Price'] * 10  # Scale to similar range as efficiency/reliability
        )
        
        return self.df
    
    def adjust_weights_from_priorities(self, priorities):
        """
        Adjust weights based on user priorities
        
        Parameters:
        -----------
        priorities : dict
            Dictionary with user priorities for energy_efficiency, security, ease_of_use, scalability
            Each value should be between 1-10
        
        Returns:
        --------
        dict
            Adjusted weights for composite score calculation
        """
        # Normalize priorities
        total = sum(priorities.values())
        norm_priorities = {k: v/total for k, v in priorities.items()}
        
        # Map priorities to weights
        weights = {
            "efficiency": 0.3 + (0.4 * norm_priorities.get("energy_efficiency", 0.25)),
            "reliability": 0.3 + (0.3 * norm_priorities.get("security", 0.25) + 0.1 * norm_priorities.get("ease_of_use", 0.25)),
            "price": 0.2 + (0.2 * (1 - norm_priorities.get("scalability", 0.25)))
        }
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def estimate_components_per_room(self, num_rooms, floor_area=None):
        """
        Estimate the number and types of components needed per room
        
        Parameters:
        -----------
        num_rooms : int
            Number of rooms in the house
        floor_area : float, optional
            Total floor area in square meters
        
        Returns:
        --------
        dict
            Dictionary with room allocations and component requirements
        """
        # Simple room allocation based on number of rooms
        room_allocation = {}
        
        if num_rooms <= 2:  # Studio or 1BR
            room_allocation = {
                "Living Room": 1,
                "Bedroom": 0 if num_rooms == 1 else 1,
                "Kitchen": 1,
                "Bathroom": 1,
                "Hallway": 0,
                "Entrance": 1
            }
        elif num_rooms <= 4:  # 2BR
            room_allocation = {
                "Living Room": 1,
                "Bedroom": 2,
                "Kitchen": 1,
                "Bathroom": 1,
                "Hallway": 1,
                "Entrance": 1
            }
        elif num_rooms <= 6:  # 3BR
            room_allocation = {
                "Living Room": 1,
                "Bedroom": 3,
                "Kitchen": 1,
                "Bathroom": 2,
                "Hallway": 1,
                "Entrance": 1
            }
        else:  # 4+BR
            room_allocation = {
                "Living Room": 1,
                "Bedroom": num_rooms - 5,
                "Kitchen": 1,
                "Bathroom": 2,
                "Hallway": 1,
                "Entrance": 1
            }
        
        # Calculate component requirements based on room allocation
        component_requirements = defaultdict(int)
        for room_type, count in room_allocation.items():
            if count > 0 and room_type in self.room_types:
                for category, num_components in self.room_types[room_type].items():
                    component_requirements[category] += num_components * count
        
        return {
            "room_allocation": room_allocation,
            "component_requirements": dict(component_requirements)
        }
    
    def optimize_component_selection(self, budget, component_requirements, weights=None):
        """
        Optimize component selection based on budget and requirements
        
        Parameters:
        -----------
        budget : float
            Total budget in INR
        component_requirements : dict
            Dictionary with required number of components per category
        weights : dict, optional
            Weights for composite score calculation
        
        Returns:
        --------
        dict
            Dictionary with selected components and their details
        """
        # Calculate composite scores if not already done
        if 'Composite_Score' not in self.df.columns:
            self.calculate_composite_score(weights)
        
        # Create optimization problem
        prob = LpProblem("SmartHomeOptimization", LpMaximize)
        
        # Create decision variables (1 if component is selected, 0 otherwise)
        component_vars = {}
        for idx, row in self.df.iterrows():
            component_vars[idx] = LpVariable(f"component_{idx}", 0, None, LpInteger)
        
        # Objective function: maximize composite score
        prob += lpSum([component_vars[idx] * row['Composite_Score'] for idx, row in self.df.iterrows()])
        
        # Budget constraint
        prob += lpSum([component_vars[idx] * row['Price_INR'] for idx, row in self.df.iterrows()]) <= budget
        
        # Component requirements constraints
        for category, required_count in component_requirements.items():
            prob += lpSum([component_vars[idx] for idx, row in self.df.iterrows() if row['Category'] == category]) >= required_count
        
        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False))
        
        # Extract results
        selected_components = []
        for idx, var in component_vars.items():
            if var.value() is not None and var.value() > 0:
                component = self.df.iloc[idx].to_dict()
                component['Quantity'] = int(var.value())
                selected_components.append(component)
        
        # Calculate total cost
        total_cost = sum(comp['Price_INR'] * comp['Quantity'] for comp in selected_components)
        
        return {
            "selected_components": selected_components,
            "total_cost": total_cost,
            "remaining_budget": budget - total_cost,
            "status": LpStatus[prob.status]
        }
    
    def generate_room_allocation(self, selected_components, room_allocation):
        """
        Allocate selected components to rooms
        
        Parameters:
        -----------
        selected_components : list
            List of selected components with quantities
        room_allocation : dict
            Dictionary with room types and counts
        
        Returns:
        --------
        dict
            Dictionary with room allocations of components
        """
        # Create a copy of components to work with
        components_to_allocate = []
        for comp in selected_components:
            for _ in range(comp['Quantity']):
                components_to_allocate.append(comp.copy())
                components_to_allocate[-1]['Quantity'] = 1  # Set quantity to 1 for individual allocation
        
        # Create room instances based on allocation
        rooms = []
        for room_type, count in room_allocation.items():
            for i in range(count):
                rooms.append({
                    "name": f"{room_type} {i+1 if count > 1 else ''}".strip(),
                    "type": room_type,
                    "components": []
                })
        
        # Sort components by category for easier allocation
        components_by_category = defaultdict(list)
        for comp in components_to_allocate:
            components_by_category[comp['Category']].append(comp)
        
        # Allocate components to rooms based on room type preferences
        for room in rooms:
            room_type = room['type']
            if room_type in self.room_types:
                for category, count in self.room_types[room_type].items():
                    # Get components of this category
                    category_components = components_by_category[category]
                    # Allocate up to the required count for this room type
                    for _ in range(min(count, len(category_components))):
                        if category_components:
                            room['components'].append(category_components.pop(0))
        
        # Allocate any remaining components to rooms that can use them
        for category, components in components_by_category.items():
            for comp in components:
                # Find rooms that use this category
                suitable_rooms = [r for r in rooms if category in self.room_types.get(r['type'], {})]
                if suitable_rooms:
                    # Sort rooms by number of components of this category they already have
                    suitable_rooms.sort(key=lambda r: len([c for c in r['components'] if c['Category'] == category]))
                    # Allocate to the room with the fewest components of this category
                    suitable_rooms[0]['components'].append(comp)
                else:
                    # If no suitable room, allocate to the room with the fewest components overall
                    rooms.sort(key=lambda r: len(r['components']))
                    rooms[0]['components'].append(comp)
        
        return rooms
    
    def generate_configuration(self, num_rooms, budget, priorities, floor_area=None):
        """
        Generate a complete smart home configuration based on user inputs
        
        Parameters:
        -----------
        num_rooms : int
            Number of rooms in the house
        budget : float
            Total budget in INR
        priorities : dict
            Dictionary with user priorities (energy_efficiency, security, ease_of_use, scalability)
        floor_area : float, optional
            Total floor area in square meters
        
        Returns:
        --------
        dict
            Complete configuration with room allocations and component details
        """
        # Adjust weights based on user priorities
        weights = self.adjust_weights_from_priorities(priorities)
        
        # Calculate composite scores
        self.calculate_composite_score(weights)
        
        # Estimate component requirements
        requirements = self.estimate_components_per_room(num_rooms, floor_area)
        room_allocation = requirements['room_allocation']
        component_requirements = requirements['component_requirements']
        
        # Optimize component selection based on budget and requirements
        optimization_result = self.optimize_component_selection(budget, component_requirements, weights)
        
        # Allocate components to rooms
        room_allocations = self.generate_room_allocation(optimization_result['selected_components'], room_allocation)
        
        # Return complete configuration
        return {
            "room_allocations": room_allocations,
            "optimization_result": optimization_result,
            "weights": weights,
            "priorities": priorities,
            "budget": budget,
            "total_cost": optimization_result['total_cost'],
            "remaining_budget": optimization_result['remaining_budget']
        }
    
    def visualize_component_distribution(self, configuration):
        """
        Visualize the distribution of components by category
        
        Parameters:
        -----------
        configuration : dict
            Configuration generated by generate_configuration method
        """
        # Extract selected components
        selected_components = configuration['optimization_result']['selected_components']
        
        # Count components by category
        category_counts = defaultdict(int)
        for comp in selected_components:
            category_counts[comp['Category']] += comp['Quantity']
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        # Sort by count
        sorted_indices = np.argsort(counts)
        categories = [categories[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Plot
        bars = plt.barh(categories, counts, color=plt.cm.viridis(np.linspace(0, 0.8, len(categories))))
        plt.xlabel('Number of Components')
        plt.ylabel('Category')
        plt.title('Component Distribution by Category')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.0f}', 
                    ha='left', va='center')
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_cost_breakdown(self, configuration):
        """
        Visualize the cost breakdown by category
        
        Parameters:
        -----------
        configuration : dict
            Configuration generated by generate_configuration method
        """
        # Extract selected components
        selected_components = configuration['optimization_result']['selected_components']
        
        # Calculate cost by category
        category_costs = defaultdict(float)
        for comp in selected_components:
            category_costs[comp['Category']] += comp['Price_INR'] * comp['Quantity']
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        categories = list(category_costs.keys())
        costs = list(category_costs.values())
        
        # Calculate percentages
        total_cost = sum(costs)
        percentages = [100 * cost / total_cost for cost in costs]
        
        # Plot
        plt.pie(costs, labels=categories, autopct='%1.1f%%', startangle=90,
               colors=plt.cm.viridis(np.linspace(0, 0.8, len(categories))))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(f'Cost Breakdown by Category (Total: ₹{total_cost:,.2f})')
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_room_allocation(self, configuration):
        """
        Visualize the component allocation by room
        
        Parameters:
        -----------
        configuration : dict
            Configuration generated by generate_configuration method
        """
        # Extract room allocations
        room_allocations = configuration['room_allocations']
        
        # Count components by room and category
        room_category_counts = {}
        for room in room_allocations:
            room_name = room['name']
            category_counts = defaultdict(int)
            for comp in room['components']:
                category_counts[comp['Category']] += 1
            room_category_counts[room_name] = dict(category_counts)
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 8))
        
        # Get all unique categories
        all_categories = set()
        for counts in room_category_counts.values():
            all_categories.update(counts.keys())
        all_categories = sorted(list(all_categories))
        
        # Prepare data for stacked bars
        rooms = list(room_category_counts.keys())
        bottoms = np.zeros(len(rooms))
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_categories)))
        
        # Plot each category as a segment of the stacked bar
        for i, category in enumerate(all_categories):
            category_counts = [room_category_counts[room].get(category, 0) for room in rooms]
            plt.bar(rooms, category_counts, bottom=bottoms, label=category, color=colors[i])
            bottoms += category_counts
        
        plt.xlabel('Room')
        plt.ylabel('Number of Components')
        plt.title('Component Allocation by Room')
        plt.legend(title='Category')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return plt.gcf()
    
    def generate_report(self, configuration, output_file=None):
        """
        Generate a comprehensive report of the smart home configuration
        
        Parameters:
        -----------
        configuration : dict
            Configuration generated by generate_configuration method
        output_file : str, optional
            Path to save the report as HTML
        
        Returns:
        --------
        str
            HTML report if output_file is None, otherwise None
        """
        # Extract data from configuration
        room_allocations = configuration['room_allocations']
        total_cost = configuration['total_cost']
        remaining_budget = configuration['remaining_budget']
        budget = configuration['budget']
        priorities = configuration['priorities']
        
        # Create HTML report
        html = ["<html><head><style>",
                "body { font-family: Arial, sans-serif; margin: 20px; }",
                "h1, h2, h3 { color: #2c3e50; }",
                "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
                "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                "th { background-color: #f2f2f2; }",
                "tr:nth-child(even) { background-color: #f9f9f9; }",
                "tr:hover { background-color: #f5f5f5; }",
                ".summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
                ".room { background-color: #f0f7fa; padding: 10px; margin-bottom: 15px; border-radius: 5px; }",
                "</style></head><body>",
                "<h1>Smart Home Configuration Report</h1>"]
        
        # Add summary section
        html.append("<div class='summary'>")
        html.append(f"<h2>Summary</h2>")
        html.append(f"<p><strong>Total Budget:</strong> ₹{budget:,.2f}</p>")
        html.append(f"<p><strong>Total Cost:</strong> ₹{total_cost:,.2f} ({total_cost/budget*100:.1f}% of budget)</p>")
        html.append(f"<p><strong>Remaining Budget:</strong> ₹{remaining_budget:,.2f}</p>")
        
        # Add priorities section
        html.append("<h3>User Priorities</h3>")
        html.append("<ul>")
        for priority, value in priorities.items():
            html.append(f"<li><strong>{priority.replace('_', ' ').title()}:</strong> {value}/10</li>")
        html.append("</ul>")
        html.append("</div>")
        
        # Add room allocation section
        html.append("<h2>Room Allocations</h2>")
        
        for room in room_allocations:
            html.append(f"<div class='room'>")
            html.append(f"<h3>{room['name']}</h3>")
            
            # Count components by category
            category_counts = defaultdict(int)
            for comp in room['components']:
                category_counts[comp['Category']] += 1
            
            # Add category summary
            html.append("<p><strong>Components:</strong> ")
            html.append(", ".join([f"{count} {category}" for category, count in category_counts.items()]))
            html.append("</p>")
            
            # Add component table
            html.append("<table>")
            html.append("<tr><th>Component</th><th>Category</th><th>Price (₹)</th><th>Efficiency</th><th>Reliability</th></tr>")
            
            # Sort components by category and name
            sorted_components = sorted(room['components'], key=lambda x: (x['Category'], x['Component_Name']))
            
            for comp in sorted_components:
                html.append(f"<tr>")
                html.append(f"<td>{comp['Component_Name']}</td>")
                html.append(f"<td>{comp['Category']}</td>")
                html.append(f"<td>{comp['Price_INR']:,.2f}</td>")
                html.append(f"<td>{comp['Efficiency']:.1f}/10</td>")
                html.append(f"<td>{comp['Reliability']:.1f}/10</td>")
                html.append(f"</tr>")
            
            html.append("</table>")
            html.append("</div>")
        
        # Close HTML
        html.append("</body></html>")
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(html))
            return None
        
        return "\n".join(html)