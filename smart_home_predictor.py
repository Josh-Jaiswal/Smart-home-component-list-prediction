import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
import random
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

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
        
        # Initialize ML models
        self.initialize_ml_models()
        
        # Train models with available data
        self.train_models()
    
    def initialize_ml_models(self):
        """Initialize machine learning models for component selection and optimization"""
        try:
            # Import ML libraries
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.neighbors import KNeighborsRegressor
            import xgboost as xgb
            from sklearn.neural_network import MLPRegressor
            from sklearn.tree import DecisionTreeClassifier
            
            # Create models directory if it doesn't exist
            import os
            os.makedirs('models', exist_ok=True)
            
            # Check if models already exist and load them
            if os.path.exists('models/compatibility_model.pkl'):
                print("Loading pre-trained models...")
                try:
                    self.compatibility_model = joblib.load('models/compatibility_model.pkl')
                    self.performance_model = joblib.load('models/performance_model.pkl')
                    self.nn_model = joblib.load('models/nn_model.pkl')
                    self.kmeans_model = joblib.load('models/kmeans_model.pkl')
                    self.knn_model = joblib.load('models/knn_model.pkl')
                    self.decision_tree = joblib.load('models/decision_tree.pkl')
                    self.gb_model = joblib.load('models/gb_model.pkl')
                    self.scaler = joblib.load('models/scaler.pkl')
                    print("Pre-trained models loaded successfully")
                    return True
                except Exception as e:
                    print(f"Error loading models: {e}")
                    print("Will train new models")
            
            # Random Forest for component compatibility prediction
            self.compatibility_model = RandomForestClassifier(
                n_estimators=50,  # Reduced from 100
                max_depth=8,      # Reduced from 10
                random_state=42
            )
            
            # XGBoost for performance metrics prediction
            self.performance_model = xgb.XGBRegressor(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.1,
                max_depth=4,      # Reduced from 5
                random_state=42
            )
            
            # Neural Network for complex pattern recognition
            self.nn_model = MLPRegressor(
                hidden_layer_sizes=(32, 16),  # Reduced from (64, 32)
                activation='relu',
                solver='adam',
                max_iter=500,     # Reduced from 1000
                random_state=42
            )
            
            # K-Means for component clustering
            self.kmeans_model = KMeans(
                n_clusters=4,     # Reduced from 5
                random_state=42
            )
            
            # KNN for similar component recommendations
            self.knn_model = KNeighborsRegressor(
                n_neighbors=3,    # Reduced from 5
                weights='distance'
            )
            
            # Decision Tree for rule extraction
            self.decision_tree = DecisionTreeClassifier(
                max_depth=4,      # Reduced from 5
                random_state=42
            )
            
            # Gradient Boosting for score refinement
            self.gb_model = GradientBoostingRegressor(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Feature scaler for preprocessing
            self.scaler = StandardScaler()
            
            # Train models with available data
            self._train_initial_models()
            
            return True
        except ImportError as e:
            print(f"Warning: ML libraries not available - {e}")
            print("Continuing with rule-based approach only")
            return False
    
    def _train_initial_models(self):
        """Train ML models with available component data"""
        try:
            # Extract features for training
            features = self._extract_component_features()
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create synthetic targets for training (since we don't have user feedback)
            # For compatibility: create synthetic compatibility scores based on component attributes
            compatibility_targets = self._create_synthetic_compatibility_targets()
            
            # For performance: use efficiency as a proxy for performance
            performance_targets = self.df['Efficiency'].values
            
            # Train models
            # 1. Random Forest for compatibility
            self.compatibility_model.fit(scaled_features, compatibility_targets)
            
            # 2. XGBoost for performance prediction
            self.performance_model.fit(scaled_features, performance_targets)
            
            # 3. Neural Network for complex patterns
            self.nn_model.fit(scaled_features, self.df['Efficiency'].values)
            
            # 4. K-Means for component clustering
            self.kmeans_model.fit(scaled_features)
            # Add cluster labels to dataframe
            self.df['Cluster'] = self.kmeans_model.labels_
            
            # 5. KNN for similar component recommendations
            self.knn_model.fit(scaled_features, self.df['Price_INR'].values)
            
            # 6. Decision Tree for rule extraction
            self.decision_tree.fit(scaled_features, compatibility_targets)
            
            # 7. Gradient Boosting for score refinement
            self.gb_model.fit(scaled_features, self.df['Reliability'].values)
            
            print("ML models trained successfully")
            
        except Exception as e:
            print(f"Error training ML models: {e}")
            print("Continuing with rule-based approach")
    
    def _extract_component_features(self):
        """Extract features from components for ML models"""
        # Select numerical features
        numerical_features = ['Price_INR', 'Efficiency', 'Reliability']
        
        # Create feature matrix
        features = self.df[numerical_features].copy()
        
        # Add one-hot encoded features for categories
        for category in self.df['Category'].unique():
            features[f'Category_{category}'] = (self.df['Category'] == category).astype(int)
        
        return features
    
    def _create_synthetic_compatibility_targets(self):
        """Create synthetic compatibility targets for training"""
        # This is a simplified approach - in a real scenario, you'd use actual compatibility data
        # Here we'll use a heuristic: components with similar efficiency and reliability are more compatible
        
        # Normalize efficiency and reliability
        efficiency_norm = (self.df['Efficiency'] - self.df['Efficiency'].min()) / (self.df['Efficiency'].max() - self.df['Efficiency'].min())
        reliability_norm = (self.df['Reliability'] - self.df['Reliability'].min()) / (self.df['Reliability'].max() - self.df['Reliability'].min())
        
        # Create a synthetic compatibility score (0 or 1)
        # Components with both high efficiency and reliability are considered "compatible"
        compatibility = ((efficiency_norm + reliability_norm) / 2 > 0.6).astype(int)
        
        return compatibility
    
    def enhance_component_scores_with_ml(self, components, user_features=None, model_subset='all'):
        """
        Enhance component scores using ML models
        
        Parameters:
        -----------
        components : list
            List of components to score
        user_features : list, optional
            User features for personalization
        model_subset : str, optional
            Which subset of models to use ('all', 'minimal', 'performance')
        
        Returns:
        --------
        list
            Components with enhanced scores
        """
        try:
            # If ML models aren't initialized, return original components
            if not hasattr(self, 'compatibility_model'):
                return components
            
            # Extract features for each component
            component_features = []
            for comp in components:
                # Basic numerical features
                features = [
                    comp['Price_INR'],
                    comp['Efficiency'],
                    comp['Reliability']
                ]
                
                # One-hot encode category
                for category in self.df['Category'].unique():
                    features.append(1 if comp['Category'] == category else 0)
                
                component_features.append(features)
            
            # Scale features
            scaled_features = self.scaler.transform(component_features)
            
            # Select which models to use based on subset parameter
            if model_subset == 'minimal':
                # Use only the most essential models
                compatibility_scores = self.compatibility_model.predict_proba(scaled_features)[:, 1]
                performance_scores = np.ones(len(components)) * 0.5  # Default value
                
                # Calculate ML-based score with just compatibility
                ml_scores = compatibility_scores
                
            elif model_subset == 'performance':
                # Focus on performance-related models
                compatibility_scores = np.ones(len(components)) * 0.5  # Default value
                performance_scores = self.performance_model.predict(scaled_features)
                
                # Normalize performance scores
                if len(performance_scores) > 0:
                    min_score = min(performance_scores)
                    max_score = max(performance_scores)
                    if max_score > min_score:
                        performance_scores = (performance_scores - min_score) / (max_score - min_score)
                
                # Calculate ML-based score with just performance
                ml_scores = performance_scores
                
            else:  # 'all' or any other value
                # Use all available models
                compatibility_scores = self.compatibility_model.predict_proba(scaled_features)[:, 1]
                performance_scores = self.performance_model.predict(scaled_features)
                
                # Normalize performance scores
                if len(performance_scores) > 0:
                    min_score = min(performance_scores)
                    max_score = max(performance_scores)
                    if max_score > min_score:
                        performance_scores = (performance_scores - min_score) / (max_score - min_score)
                
                # Calculate ML-based score with all models
                ml_scores = 0.5 * compatibility_scores + 0.5 * performance_scores
            
            # Enhance component scores
            for i, comp in enumerate(components):
                # Blend with original score (70% original, 30% ML)
                if 'Score' in comp:
                    comp['Score'] = 0.7 * comp['Score'] + 0.3 * ml_scores[i]
                else:
                    comp['Score'] = ml_scores[i]
            
            return components
        
        except Exception as e:
            print(f"Error enhancing scores with ML: {e}")
            return components
    
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
    
    def train_models(self):
        """Train ML models with available component data"""
        try:
            # Extract features for training
            features = self._extract_component_features()
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create synthetic targets for training (since we don't have user feedback)
            # For compatibility: create synthetic compatibility scores based on component attributes
            compatibility_targets = self._create_synthetic_compatibility_targets()
            
            # For performance: use efficiency as a proxy for performance
            performance_targets = self.df['Efficiency'].values
            
            # Train models
            # 1. Random Forest for compatibility
            self.compatibility_model.fit(scaled_features, compatibility_targets)
            
            # 2. XGBoost for performance prediction
            self.performance_model.fit(scaled_features, performance_targets)
            
            # 3. Neural Network for complex patterns
            self.nn_model.fit(scaled_features, self.df['Composite_Score'].values if 'Composite_Score' in self.df.columns 
                             else self.df['Efficiency'].values)
            
            # 4. K-Means for component clustering
            self.kmeans_model.fit(scaled_features)
            # Add cluster labels to dataframe
            self.df['Cluster'] = self.kmeans_model.labels_
            
            # 5. KNN for similar component recommendations
            self.knn_model.fit(scaled_features, self.df['Price_INR'].values)
            
            # 6. Decision Tree for rule extraction
            self.decision_tree.fit(scaled_features, compatibility_targets)
            
            # 7. Gradient Boosting for score refinement
            self.gb_model.fit(scaled_features, self.df['Reliability'].values)
            
            print("ML models trained successfully")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            print(f"Error training ML models: {e}")
            print("Continuing with rule-based approach")
    
    def _extract_component_features(self):
        """Extract features from components for ML models"""
        # Select numerical features
        numerical_features = ['Price_INR', 'Efficiency', 'Reliability']
        
        # One-hot encode categorical features
        categorical_features = ['Category']
        
        # Create feature matrix
        features = self.df[numerical_features].copy()
        
        # Add one-hot encoded features
        for category in self.df['Category'].unique():
            features[f'Category_{category}'] = (self.df['Category'] == category).astype(int)
        
        return features
    
    def _create_synthetic_compatibility_targets(self):
        """Create synthetic compatibility targets for training"""
        # This is a simplified approach - in a real scenario, you'd use actual compatibility data
        # Here we'll use a heuristic: components with similar efficiency and reliability are more compatible
        
        # Normalize efficiency and reliability
        efficiency_norm = (self.df['Efficiency'] - self.df['Efficiency'].min()) / (self.df['Efficiency'].max() - self.df['Efficiency'].min())
        reliability_norm = (self.df['Reliability'] - self.df['Reliability'].min()) / (self.df['Reliability'].max() - self.df['Reliability'].min())
        
        # Create a synthetic compatibility score (0 or 1)
        # Components with both high efficiency and reliability are considered "compatible"
        compatibility = ((efficiency_norm + reliability_norm) / 2 > 0.6).astype(int)
        
        return compatibility
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.compatibility_model, 'models/compatibility_model.pkl')
            joblib.dump(self.performance_model, 'models/performance_model.pkl')
            joblib.dump(self.nn_model, 'models/nn_model.pkl')
            joblib.dump(self.kmeans_model, 'models/kmeans_model.pkl')
            joblib.dump(self.knn_model, 'models/knn_model.pkl')
            joblib.dump(self.decision_tree, 'models/decision_tree.pkl')
            joblib.dump(self.gb_model, 'models/gb_model.pkl')
            joblib.dump(self.scaler, 'models/scaler.pkl')
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def calculate_composite_score(self, weights=None):
        """
        Calculate composite score for each component based on weights and ML predictions
        
        Parameters:
        -----------
        weights : dict
            Dictionary with weights for efficiency, reliability, and price
        
        Returns:
        --------
        DataFrame with added composite score column
        """
        # Default weights if not provided
        if weights is None:
            weights = {
                "efficiency": 0.3,
                "reliability": 0.3,
                "price": 0.4
            }
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Calculate price score (inverse of normalized price)
        price_max = self.df['Price_INR'].max()
        price_min = self.df['Price_INR'].min()
        self.df['Price_Score'] = 1 - ((self.df['Price_INR'] - price_min) / (price_max - price_min))
        
        # Calculate rule-based composite score
        self.df['Rule_Score'] = (
            weights['efficiency'] * self.df['Efficiency'] / 10 +
            weights['reliability'] * self.df['Reliability'] / 10 +
            weights['price'] * self.df['Price_Score']
        )
        
        # Try to enhance with ML models if available
        try:
            if hasattr(self, 'compatibility_model') and hasattr(self, 'performance_model'):
                # Extract features for ML prediction
                features = self._extract_component_features()
                scaled_features = self.scaler.transform(features)
                
                # Get ML predictions
                compatibility_scores = self.compatibility_model.predict_proba(scaled_features)[:, 1]
                performance_scores = self.performance_model.predict(scaled_features)
                
                # Normalize performance scores
                performance_scores = (performance_scores - performance_scores.min()) / (performance_scores.max() - performance_scores.min() + 1e-10)
                
                # Calculate ML-based score
                self.df['ML_Score'] = 0.5 * compatibility_scores + 0.5 * performance_scores
                
                # Blend rule-based and ML-based scores (70% rule, 30% ML)
                self.df['Composite_Score'] = 0.7 * self.df['Rule_Score'] + 0.3 * self.df['ML_Score']
            else:
                # If ML models aren't available, use rule-based score
                self.df['Composite_Score'] = self.df['Rule_Score']
        except Exception as e:
            print(f"Error in ML scoring: {e}")
            # Fallback to rule-based score
            self.df['Composite_Score'] = self.df['Rule_Score']
        
        # Add a small random factor to break ties (0.5% variation)
        import random
        self.df['Composite_Score'] = self.df['Composite_Score'] * [random.uniform(0.995, 1.005) for _ in range(len(self.df))]
        
        return self.df['Composite_Score']
    
    def adjust_weights_from_priorities(self, priorities, variant=0):
        """
        Adjust component selection weights based on user priorities
        
        Parameters:
        -----------
        priorities : dict
            Dictionary with user priorities (energy_efficiency, security, ease_of_use, scalability)
        variant : int, optional
            Variant number to generate different configurations
            
        Returns:
        --------
        dict
            Dictionary with adjusted weights for component selection
        """
        # Normalize priorities to sum to 1
        total_priority = sum(priorities.values())
        norm_priorities = {k: v/total_priority for k, v in priorities.items()}
        
        # Base weights
        weights = {
            "efficiency": 0.25 + (norm_priorities['energy_efficiency'] * 0.05),
            "reliability": 0.25 + (norm_priorities['security'] * 0.05),
            "price": 0.5 - ((norm_priorities['energy_efficiency'] + norm_priorities['security']) * 0.025)
        }
        
        # Apply variations based on variant number
        if variant == 1:
            # Energy-Efficient: Strongly favor efficiency
            weights['efficiency'] = min(0.6, weights['efficiency'] * 1.8)
            weights['price'] = max(0.2, weights['price'] * 0.7)
        elif variant == 2:
            # High-Reliability: Strongly favor reliability
            weights['reliability'] = min(0.6, weights['reliability'] * 1.8)
            weights['price'] = max(0.2, weights['price'] * 0.7)
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def generate_multiple_configurations(self, num_rooms, budget, priorities):
        """
        Generate multiple smart home configurations based on user inputs
        
        Parameters:
        -----------
        num_rooms : int
            Number of rooms to configure
        budget : float
            Total budget for the smart home setup
        priorities : dict
            Dictionary with user priorities (energy_efficiency, security, ease_of_use, scalability)
            
        Returns:
        --------
        list
            List of configuration dictionaries
        """
        configurations = []
            # Configuration names and descriptions
        config_names = [
        "Balanced Setup",
        "Energy-Efficient Setup",
        "High-Security Setup"
        ]
    
        config_descriptions = [
        "A balanced configuration that optimizes for all priorities",
        "Prioritizes energy efficiency with smart power management",
        "Focuses on security features and reliability"
        ]
        # Generate 3 different configurations with different weight variations
        for variant in range(3):
            # Adjust weights based on priorities and variant
            weights = self.adjust_weights_from_priorities(priorities, variant)
            
            # Calculate composite scores with the adjusted weights
            self.calculate_composite_score(weights)
            
            # Optimize component selection
            optimization_result = self.optimize_component_selection(num_rooms, budget, weights)
            
            # Allocate components to rooms
            room_allocations = self.allocate_components_to_rooms(optimization_result['selected_components'], num_rooms)
            
            # Create configuration dictionary
            configuration = {
                'variant': variant,
                'name': config_names[variant],
                'description': config_descriptions[variant],
                'weights': weights,
                'optimization_result': optimization_result,
                'room_allocations': room_allocations,
                'total_cost': optimization_result['total_cost'],
                'total_components': len(optimization_result['selected_components'])
            }
            
            configurations.append(configuration)
        
        return configurations
    
    def optimize_component_selection(self, num_rooms, budget, weights=None):
        """
        Optimize component selection based on budget and weights
        
        Parameters:
        -----------
        num_rooms : int
            Number of rooms to configure
        budget : float
            Total budget for the smart home setup
        weights : dict, optional
            Dictionary with weights for efficiency, reliability, and price
            
        Returns:
        --------
        dict
            Dictionary with optimization results
        """
        # Calculate composite scores if weights are provided
        if weights:
            self.calculate_composite_score(weights)
        
        # Create a copy of the dataframe for optimization
        df_opt = self.df.copy()
        
        # Determine required components based on room types
        required_components = self._determine_required_components(num_rooms)
        
        # Select components based on composite score and budget
        selected_components = []
        total_cost = 0
        
        # First, select required components for each category
        for category, count in required_components.items():
            # Get components in this category, sorted by composite score
            category_components = df_opt[df_opt['Category'] == category].sort_values('Composite_Score', ascending=False)
            
            # Select the top N components based on required count
            for i in range(min(count, len(category_components))):
                component = category_components.iloc[i].to_dict()
                
                # Check if adding this component exceeds the budget
                if total_cost + component['Price_INR'] <= budget:
                    selected_components.append(component)
                    total_cost += component['Price_INR']
        
        # Try to enhance component scores with ML if available
        try:
            if hasattr(self, 'compatibility_model'):
                selected_components = self.enhance_component_scores_with_ml(selected_components)
        except Exception as e:
            print(f"Error enhancing component scores: {e}")
        
        # Return optimization result
        return {
            'selected_components': selected_components,
            'total_cost': total_cost,
            'budget_utilization': total_cost / budget if budget > 0 else 0
        }
    
    def _determine_required_components(self, num_rooms):
        """
        Determine required components based on number of rooms
        
        Parameters:
        -----------
        num_rooms : int
            Number of rooms to configure
            
        Returns:
        --------
        dict
            Dictionary with required component counts by category
        """
        # Initialize required components dictionary
        required_components = {
            "Lighting": 0,
            "Security": 0,
            "HVAC": 0,
            "Energy Management": 0
        }
        
        # Assign room types based on number of rooms
        room_types = []
        if num_rooms >= 1:
            room_types.append("Living Room")
        if num_rooms >= 2:
            room_types.append("Bedroom")
        if num_rooms >= 3:
            room_types.append("Kitchen")
        if num_rooms >= 4:
            room_types.append("Bathroom")
        if num_rooms >= 5:
            room_types.append("Hallway")
        if num_rooms >= 6:
            room_types.append("Entrance")
        
        # Add additional bedrooms if needed
        while len(room_types) < num_rooms:
            room_types.append("Bedroom")
        
        # Calculate required components based on room types
        for room_type in room_types:
            if room_type in self.room_types:
                for category, count in self.room_types[room_type].items():
                    required_components[category] += count
        
        return required_components
    
    def allocate_components_to_rooms(self, components, num_rooms):
        """
        Allocate components to rooms based on room types
        
        Parameters:
        -----------
        components : list
            List of selected components
        num_rooms : int
            Number of rooms to configure
            
        Returns:
        --------
        list
            List of room dictionaries with allocated components
        """
        # Assign room types based on number of rooms
        room_types = []
        if num_rooms >= 1:
            room_types.append("Living Room")
        if num_rooms >= 2:
            room_types.append("Bedroom")
        if num_rooms >= 3:
            room_types.append("Kitchen")
        if num_rooms >= 4:
            room_types.append("Bathroom")
        if num_rooms >= 5:
            room_types.append("Hallway")
        if num_rooms >= 6:
            room_types.append("Entrance")
        
        # Add additional bedrooms if needed
        while len(room_types) < num_rooms:
            room_types.append(f"Bedroom {len([r for r in room_types if 'Bedroom' in r]) + 1}")
        
        # Initialize rooms with empty component lists
        rooms = [{'name': room_type, 'components': []} for room_type in room_types]
        
        # Group components by category
        components_by_category = {}
        for component in components:
            category = component['Category']
            if category not in components_by_category:
                components_by_category[category] = []
            components_by_category[category].append(component)
        
        # Allocate components to rooms based on room types
        for i, room in enumerate(rooms):
            room_type = room['name'].split(' ')[0] if ' ' in room['name'] else room['name']
            
            if room_type in self.room_types:
                # Allocate components based on room type requirements
                for category, count in self.room_types[room_type].items():
                    if category in components_by_category and components_by_category[category]:
                        # Allocate up to 'count' components of this category to the room
                        for j in range(min(count, len(components_by_category[category]))):
                            if components_by_category[category]:
                                component = components_by_category[category].pop(0)
                                room['components'].append(component)
        
        # Allocate any remaining components to rooms that can use them
        for category, category_components in components_by_category.items():
            for component in category_components:
                # Find a room that can use this component
                for room in rooms:
                    room_type = room['name'].split(' ')[0] if ' ' in room['name'] else room['name']
                    if room_type in self.room_types and self.room_types[room_type].get(category, 0) > 0:
                        room['components'].append(component)
                        break
        
        return rooms
    
    def visualize_component_distribution(self, configuration):
        """
        Visualize component distribution by category
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with component distribution visualization
        """
        # Extract component categories
        components = configuration['optimization_result']['selected_components']
        categories = [comp['Category'] for comp in components]
        
        # Count components by category
        category_counts = {}
        for category in categories:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bar chart
        bars = ax.bar(category_counts.keys(), category_counts.values())
        
        # Add labels and title
        ax.set_xlabel('Component Category')
        ax.set_ylabel('Number of Components')
        ax.set_title('Component Distribution by Category')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_cost_breakdown(self, configuration):
        """
        Visualize cost breakdown by category
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with cost breakdown visualization
        """
        # Extract component categories and prices
        components = configuration['optimization_result']['selected_components']
        
        # Calculate cost by category
        category_costs = {}
        for comp in components:
            category = comp['Category']
            price = comp['Price_INR']
            
            if category in category_costs:
                category_costs[category] += price
            else:
                category_costs[category] = price
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            category_costs.values(), 
            labels=category_costs.keys(),
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Add title
        ax.set_title('Cost Breakdown by Category')
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_room_allocation(self, configuration):
        """
        Visualize component allocation by room
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with room allocation visualization
        """
        # Extract room allocations
        rooms = configuration.get('room_allocations', [])
        
        # Calculate component counts by room
        room_names = []
        component_counts = []
        
        for room in rooms:
            room_names.append(room['name'])
            component_counts.append(len(room['components']))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart
        bars = ax.barh(room_names, component_counts)
        
        # Add labels and title
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Room')
        ax.set_title('Component Allocation by Room')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.0f}', ha='left', va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, configuration, output_path=None):
        """
        Generate HTML report for a configuration
        
        Parameters:
        -----------
        configuration : dict
            Configuration dictionary
        output_path : str, optional
            Path to save the HTML report
            
        Returns:
        --------
        str
            HTML report content
        """
        # Extract configuration details
        variant = configuration['variant']
        weights = configuration['weights']
        total_cost = configuration['total_cost']
        components = configuration['optimization_result']['selected_components']
        rooms = configuration.get('room_allocations', [])
        
        # Generate HTML content
        html = f"""
        <html>
        <head>
            <title>Smart Home Configuration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #eef7fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .room {{ background-color: #f0f7e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Smart Home Configuration Report</h1>
            
            <div class="summary">
                <h2>Configuration Summary</h2>
                <p><strong>Variant:</strong> {variant}</p>
                <p><strong>Total Cost:</strong> ₹{total_cost:.2f}</p>
                <p><strong>Total Components:</strong> {len(components)}</p>
                <p><strong>Weights Used:</strong></p>
                <ul>
                    <li>Efficiency: {weights['efficiency']:.2f}</li>
                    <li>Reliability: {weights['reliability']:.2f}</li>
                    <li>Price: {weights['price']:.2f}</li>
                </ul>
            </div>
            
            <h2>Component List</h2>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Category</th>
                    <th>Price (₹)</th>
                    <th>Efficiency</th>
                    <th>Reliability</th>
                </tr>
        """
        
        # Add component rows
        for comp in components:
            component_name = comp.get('Name', comp.get('Component', f"Component {components.index(comp)+1}"))
            
            html += f"""
                <tr>
                    <td>{component_name}</td>
                    <td>{comp['Category']}</td>
                    <td>{comp['Price_INR']:.2f}</td>
                    <td>{comp['Efficiency']}</td>
                    <td>{comp['Reliability']}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Room Allocations</h2>
        """
        
        # Add room sections
        for room in rooms:
            html += f"""
            <div class="room">
                <h3>{room['name']}</h3>
                <p><strong>Components:</strong> {len(room['components'])}</p>
                
                <table>
                    <tr>
                        <th>Name</th>
                        <th>Category</th>
                        <th>Price (₹)</th>
                        <th>Efficiency</th>
                        <th>Reliability</th>
                    </tr>
            """
            
            # Add component rows for this room
            for comp in room['components']:
                # Check if 'Name' key exists, otherwise use a default or another key
                component_name = comp.get('Name', comp.get('Component', f"Component {room['components'].index(comp)+1}"))
                
                html += f"""
                    <tr>
                        <td>{component_name}</td>
                        <td>{comp['Category']}</td>
                        <td>{comp['Price_INR']:.2f}</td>
                        <td>{comp['Efficiency']}</td>
                        <td>{comp['Reliability']}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
        
        return html