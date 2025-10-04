import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import requests
from typing import Dict, Any, List, Tuple
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

np.random.seed(42)  # Reproducible random numbers

class EnergyPredictor:
    def __init__(self):
        self.models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.features: Dict[str, List[str]] = {
            'solar': ['latitude', 'longitude', 'elevation', 'annual_irradiance', 
                     'avg_temperature', 'avg_humidity', 'cloud_cover'],
            'wind': ['latitude', 'longitude', 'elevation', 'avg_wind_speed',
                    'wind_direction', 'temperature', 'air_density'],
            'hydro': ['latitude', 'longitude', 'elevation', 'precipitation',
                     'temperature', 'slope', 'drainage_area']
        }
        self.geocoding_api = "https://geocoding-api.open-meteo.com/v1/search"

    # ------------------  GEOCODING ------------------
    def get_coordinates_from_city(self, city_name: str) -> Tuple[float, float]:
        """Convert city name to coordinates using geocoding API."""
        try:
            params = {'name': city_name, 'count': 1}
            response = requests.get(self.geocoding_api, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                return result['latitude'], result['longitude']
            else:
                raise ValueError(f"City '{city_name}' not found")
                
        except Exception as e:
            logging.error(f"Geocoding failed for '{city_name}': {e}")
            raise

    def predict_energy_city(self, city_name: str, energy_type: str, area: float = 100) -> Dict[str, Any]:
        """Predict energy potential for a city name."""
        lat, lon = self.get_coordinates_from_city(city_name)
        logging.info(f"Resolved '{city_name}' to coordinates: ({lat:.4f}, {lon:.4f})")
        return self.predict_energy(lat, lon, energy_type, area)

    # ------------------ 📡 DATA FETCHING ------------------
    def fetch_combined_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch and combine environmental data for a given location."""
        try:
            nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            nasa_params = {
                'parameters': 'ALLSKY_SFC_SW_DWN,T2M,RH2M,CLRSKY_DAYS',
                'start': '20240101',
                'end': '20241231',
                'latitude': lat,
                'longitude': lon,
                'community': 'RE',
                'format': 'JSON'
            }

            response = requests.get(nasa_url, params=nasa_params, timeout=15)
            response.raise_for_status()
            nasa_data = response.json()

            processed = self.process_nasa_format(nasa_data, lat, lon)
            return processed

        except Exception as e:
            logging.warning(f"Data fetch failed for ({lat:.2f}, {lon:.2f}) — using fallback. Error: {e}")
            return self.generate_fallback_data(lat, lon)

    def process_nasa_format(self, nasa_data: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
        """Process NASA API response into usable metrics."""
        if 'properties' not in nasa_data:
            return self.generate_fallback_data(lat, lon)

        parameters = nasa_data['properties'].get('parameter', {})

        # Extract arrays
        irr_vals = [v for v in parameters.get('ALLSKY_SFC_SW_DWN', {}).values() if v is not None]
        temp_vals = [v for v in parameters.get('T2M', {}).values() if v is not None]
        hum_vals = [v for v in parameters.get('RH2M', {}).values() if v is not None]
        clr_vals = [v for v in parameters.get('CLRSKY_DAYS', {}).values() if v is not None]

        # Calculate cloud cover ~ 1 - (clear sky days / 365)
        avg_clear_days = np.mean(clr_vals) if clr_vals else 200
        cloud_cover = max(0.0, min(1.0, 1 - avg_clear_days / 365))

        return {
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat, self.get_elevation(lat, lon)]
            },
            "metrics": {
                "annual_irradiance": np.mean(irr_vals) * 365 if irr_vals else 1600,
                "avg_temperature": np.mean(temp_vals) if temp_vals else 15,
                "avg_humidity": np.mean(hum_vals) if hum_vals else 65,
                "cloud_cover": cloud_cover
            },
            "wind_speed": 5 + (abs(lat) - 30) * 0.1,
            "precipitation": 1000 - abs(lat) * 10
        }

    def generate_fallback_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Generate default values when data API fails."""
        return {
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat, self.get_elevation(lat, lon)]
            },
            "metrics": {
                "annual_irradiance": max(800, 1600 - abs(lat) * 20),
                "avg_temperature": 20 - abs(lat) * 0.5,
                "avg_humidity": 60 + abs(lat) * 0.3,
                "cloud_cover": 0.4
            },
            "wind_speed": 5 + (abs(lat) - 30) * 0.1,
            "precipitation": 1000 - abs(lat) * 10
        }

    def get_elevation(self, lat: float, lon: float) -> float:
        """Mock elevation function (replace with real API for accuracy)."""
        return 100 + np.sin(np.radians(lat)) * 500

    # ------------------  TRAINING ------------------
    def train_energy_model(self, energy_type: str, num_samples: int = 100):
        """Train model for specific energy type using simulated global samples."""
        logging.info(f"Training {energy_type.capitalize()} model on {num_samples} samples...")

        features_list = []
        targets = []

        for i in range(num_samples):
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            data = self.fetch_combined_data(lat, lon)
            elev = data['geometry']['coordinates'][2]

            if energy_type == 'solar':
                feats = [lat, lon, elev,
                         data['metrics']['annual_irradiance'],
                         data['metrics']['avg_temperature'],
                         data['metrics']['avg_humidity'],
                         data['metrics']['cloud_cover']]
                target = data['metrics']['annual_irradiance'] * 0.15 * 0.75 * 100

            elif energy_type == 'wind':
                feats = [lat, lon, elev, data['wind_speed'], 270,
                         data['metrics']['avg_temperature'], 1.225]
                target = 0.5 * 1.225 * 100 * (data['wind_speed'] ** 3) * 0.35 * 8760 * 0.3

            elif energy_type == 'hydro':
                feats = [lat, lon, elev, data['precipitation'],
                         data['metrics']['avg_temperature'], 10, 1000]
                target = data['precipitation'] * 1000 * 9.81 * 10 * 0.8 / 31536000 * 8760

            features_list.append(feats)
            targets.append(max(0, target))

        X = np.array(features_list)
        y = np.array(targets)

        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        model.fit(X_scaled, y)

        # Evaluation
        preds = model.predict(X_scaled)
        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        logging.info(f"{energy_type.capitalize()} Model — R²: {r2:.3f}, RMSE: {rmse:.2f}")

        self.scalers[energy_type] = scaler
        self.models[energy_type] = model

    # ------------------  PREDICTION ------------------
    def predict_energy(self, lat: float, lon: float, energy_type: str, area: float = 100) -> Dict[str, Any]:
        """Predict energy potential for a location and energy type."""
        if energy_type not in self.models:
            raise ValueError(f"No trained model found for '{energy_type}'. Train or load one first.")

        data = self.fetch_combined_data(lat, lon)
        elev = data['geometry']['coordinates'][2]

        if energy_type == 'solar':
            feats = [[lat, lon, elev,
                      data['metrics']['annual_irradiance'],
                      data['metrics']['avg_temperature'],
                      data['metrics']['avg_humidity'],
                      data['metrics']['cloud_cover']]]
        elif energy_type == 'wind':
            feats = [[lat, lon, elev, data['wind_speed'], 270,
                      data['metrics']['avg_temperature'], 1.225]]
        elif energy_type == 'hydro':
            feats = [[lat, lon, elev, data['precipitation'],
                      data['metrics']['avg_temperature'], 10, 1000]]

        feats_scaled = self.scalers[energy_type].transform(feats)
        base_pred = self.models[energy_type].predict(feats_scaled)[0]

        adjusted = base_pred * (area / 100)
        suitability = min(1.0, adjusted / (area * 200))

        return {
            'predicted_energy_kwh': adjusted,
            'suitability_score': suitability,
            'confidence': 0.85,
            'input_data': data
        }

    # ------------------  SAVE / LOAD ------------------
    def save_models(self, base_path='energy_models'):
        os.makedirs(base_path, exist_ok=True)
        for etype in self.models:
            joblib.dump({
                'model': self.models[etype],
                'scaler': self.scalers[etype],
                'features': self.features[etype]
            }, f'{base_path}/{etype}_model.joblib')
        logging.info(f"All models saved to '{base_path}/'")

    def load_models(self, base_path='energy_models') -> bool:
        """Load models from disk. Returns True if all models loaded successfully."""
        success = True
        for etype in ['solar', 'wind', 'hydro']:
            try:
                model_data = joblib.load(f'{base_path}/{etype}_model.joblib')
                self.models[etype] = model_data['model']
                self.scalers[etype] = model_data['scaler']
                self.features[etype] = model_data['features']
                logging.info(f"Loaded {etype} model")
            except FileNotFoundError:
                logging.warning(f"No saved model found for {etype}")
                success = False
            except Exception as e:
                logging.error(f"Error loading {etype} model: {e}")
                success = False
        return success