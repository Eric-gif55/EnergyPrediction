from flask import Flask, render_template, request, jsonify
import sqlite3
import logging
import os
from energy_predictor import EnergyPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'energy-platform-secret-key-2024'

# Debug function to check template files
def check_templates():
    """Check if all required template files exist"""
    templates = [
        'base.html',
        'index.html', 
        'energy_selection.html',
        'engineer_services.html',
        'marketplace.html',
        'energy_selling.html'
    ]
    
    missing_templates = []
    for template in templates:
        template_path = os.path.join('templates', template)
        if not os.path.exists(template_path):
            missing_templates.append(template)
    
    return missing_templates

# Initialize the predictor
try:
    predictor = EnergyPredictor()
    # Try to load existing models, if not train new ones
    if not predictor.load_models():
        logger.info("Training new energy prediction models...")
        for etype in ['solar', 'wind', 'hydro']:
            predictor.train_energy_model(etype, num_samples=100)
        predictor.save_models()
        logger.info("Models trained successfully!")
    else:
        logger.info("Pre-trained models loaded successfully!")
except Exception as e:
    logger.error(f"Error initializing EnergyPredictor: {e}")
    predictor = None

# Initialize database
def init_db():
    try:
        conn = sqlite3.connect('energy_platform.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engineers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                specialization TEXT,
                experience_years INTEGER,
                location TEXT,
                hourly_rate DECIMAL(10,2),
                rating DECIMAL(3,2),
                bio TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS marketplace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seller_name TEXT NOT NULL,
                seller_email TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_name TEXT NOT NULL,
                description TEXT,
                price DECIMAL(10,2),
                quantity INTEGER,
                location TEXT,
                condition TEXT,
                contact_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS energy_offers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seller_name TEXT NOT NULL,
                seller_email TEXT NOT NULL,
                energy_type TEXT NOT NULL,
                capacity_kwh DECIMAL(10,2),
                price_per_kwh DECIMAL(10,4),
                location TEXT,
                availability_date DATE,
                duration_months INTEGER,
                contact_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert sample data
        sample_engineers = [
            ('John Smith', 'john.smith@email.com', 'Solar Energy', 8, 'San Francisco, CA', 85.00, 4.8, 'Certified solar engineer with 8 years of experience in residential and commercial installations.'),
            ('Maria Garcia', 'maria.garcia@email.com', 'Wind Energy', 12, 'Austin, TX', 95.00, 4.9, 'Wind turbine specialist with international project experience.'),
            ('David Chen', 'david.chen@email.com', 'Hydro Power', 15, 'Seattle, WA', 110.00, 4.7, 'Hydroelectric power plant expert with focus on small-scale systems.'),
            ('Sarah Johnson', 'sarah.johnson@email.com', 'Solar Energy', 6, 'Phoenix, AZ', 75.00, 4.6, 'Solar panel installation and maintenance specialist.'),
            ('Michael Brown', 'michael.brown@email.com', 'Wind Energy', 10, 'Chicago, IL', 90.00, 4.5, 'Wind energy consultant for urban and rural applications.')
        ]
        
        sample_marketplace = [
            ('SolarTech Inc', 'sales@solartech.com', 'solar_panels', 'Premium Solar Panel 400W', 'High-efficiency monocrystalline solar panels', 250.00, 50, 'California', 'new', 'email'),
            ('WindPower Co', 'info@windpower.com', 'wind_turbines', 'Residential Wind Turbine 2kW', 'Vertical axis wind turbine for home use', 3200.00, 10, 'Texas', 'new', 'phone'),
            ('HydroSolutions', 'contact@hydrosolutions.com', 'hydro_turbines', 'Micro Hydro Turbine 5kW', 'Compact hydro turbine for small streams', 4500.00, 5, 'Washington', 'new', 'email'),
            ('EcoStorage', 'sales@ecostorage.com', 'batteries', 'Lithium Battery 10kWh', 'Home energy storage system', 5500.00, 20, 'Nevada', 'new', 'email'),
            ('GreenInverters', 'support@greeninverters.com', 'inverters', 'Solar Inverter 5kW', 'Grid-tie inverter with monitoring', 1200.00, 15, 'Arizona', 'new', 'email')
        ]
        
        sample_energy_offers = [
            ('Green Energy Co', 'contact@greenenergy.com', 'solar', 5000.00, 0.12, 'California', '2024-02-01', 60, 'email'),
            ('Power Solutions', 'info@powersolutions.com', 'wind', 3000.00, 0.09, 'Texas', '2024-01-15', 36, 'phone'),
            ('Hydro Power Inc', 'sales@hydropower.com', 'hydro', 8000.00, 0.08, 'Washington', '2024-03-01', 48, 'email'),
            ('Solar Farms Ltd', 'contact@solarfarms.com', 'solar', 12000.00, 0.11, 'Arizona', '2024-02-15', 72, 'email')
        ]
        
        cursor.executemany('INSERT OR IGNORE INTO engineers (name, email, specialization, experience_years, location, hourly_rate, rating, bio) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', sample_engineers)
        cursor.executemany('INSERT OR IGNORE INTO marketplace (seller_name, seller_email, item_type, item_name, description, price, quantity, location, condition, contact_method) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', sample_marketplace)
        cursor.executemany('INSERT OR IGNORE INTO energy_offers (seller_name, seller_email, energy_type, capacity_kwh, price_per_kwh, location, availability_date, duration_months, contact_method) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', sample_energy_offers)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

# Initialize database
init_db()

def get_db():
    """Get database connection"""
    conn = sqlite3.connect('energy_platform.db')
    conn.row_factory = sqlite3.Row
    return conn

# Routes with improved error handling
@app.route('/')
def index():
    try:
        logger.info("Rendering index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return f"""
        <html>
            <body>
                <h1>Energy Platform - Home</h1>
                <p>Template rendering failed. Error: {e}</p>
                <a href="/energy-selection">Energy Selection</a> |
                <a href="/engineer-services">Engineers</a> |
                <a href="/marketplace">Marketplace</a> |
                <a href="/energy-selling">Sell Energy</a>
            </body>
        </html>
        """

@app.route('/energy-selection')
def energy_selection():
    try:
        logger.info("Rendering energy selection page")
        return render_template('energy_selection.html')
    except Exception as e:
        logger.error(f"Error rendering energy selection: {e}")
        return f"""
        <html>
            <body>
                <h1>Energy Selection</h1>
                <p>Template rendering failed. Error: {e}</p>
                <a href="/">Back to Home</a>
            </body>
        </html>
        """

@app.route('/engineer-services')
def engineer_services():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM engineers ORDER BY rating DESC')
        engineers = cursor.fetchall()
        conn.close()
        logger.info(f"Rendering engineer services page with {len(engineers)} engineers")
        return render_template('engineer_services.html', engineers=engineers)
    except Exception as e:
        logger.error(f"Error rendering engineer services: {e}")
        return f"""
        <html>
            <body>
                <h1>Engineer Services</h1>
                <p>Template rendering failed. Error: {e}</p>
                <a href="/">Back to Home</a>
            </body>
        </html>
        """

@app.route('/marketplace')
def marketplace():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM marketplace ORDER BY created_at DESC')
        items = cursor.fetchall()
        conn.close()
        logger.info(f"Rendering marketplace page with {len(items)} items")
        return render_template('marketplace.html', items=items)
    except Exception as e:
        logger.error(f"Error rendering marketplace: {e}")
        return f"""
        <html>
            <body>
                <h1>Marketplace</h1>
                <p>Template rendering failed. Error: {e}</p>
                <a href="/">Back to Home</a>
            </body>
        </html>
        """

@app.route('/energy-selling')
def energy_selling():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM energy_offers ORDER BY created_at DESC')
        offers = cursor.fetchall()
        conn.close()
        logger.info(f"Rendering energy selling page with {len(offers)} offers")
        return render_template('energy_selling.html', offers=offers)
    except Exception as e:
        logger.error(f"Error rendering energy selling: {e}")
        return f"""
        <html>
            <body>
                <h1>Sell Energy</h1>
                <p>Template rendering failed. Error: {e}</p>
                <a href="/">Back to Home</a>
            </body>
        </html>
        """

# Debug route to check template status
@app.route('/debug')
def debug():
    missing_templates = check_templates()
    template_status = "OK" if not missing_templates else f"MISSING: {', '.join(missing_templates)}"
    
    return f"""
    <html>
        <body>
            <h1>Debug Information</h1>
            <p><strong>Current Directory:</strong> {os.getcwd()}</p>
            <p><strong>Templates Folder:</strong> {os.path.exists('templates')}</p>
            <p><strong>Template Status:</strong> {template_status}</p>
            <p><strong>Static Folder:</strong> {os.path.exists('static')}</p>
            <p><strong>Predictor Status:</strong> {'Loaded' if predictor else 'Failed'}</p>
            <hr>
            <h2>Available Routes:</h2>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/energy-selection">Energy Selection</a></li>
                <li><a href="/engineer-services">Engineer Services</a></li>
                <li><a href="/marketplace">Marketplace</a></li>
                <li><a href="/energy-selling">Sell Energy</a></li>
            </ul>
        </body>
    </html>
    """

# API Routes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if predictor is None:
            return jsonify({'success': False, 'error': 'Energy predictor not initialized'}), 500
            
        data = request.get_json()
        energy_type = data.get('energy_type', 'solar')
        area = float(data.get('area', 100))
        
        if data.get('location_type') == 'coordinates':
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            result = predictor.predict_energy(lat, lon, energy_type, area)
            location_info = f"Coordinates ({lat:.4f}, {lon:.4f})"
        else:
            city_name = data['city_name']
            result = predictor.predict_energy_city(city_name, energy_type, area)
            location_info = city_name
        
        return jsonify({
            'success': True,
            'location': location_info,
            'energy_type': energy_type,
            'area': area,
            'prediction': {
                'energy_kwh': round(result['predicted_energy_kwh'], 2),
                'suitability_score': round(result['suitability_score'], 3),
                'confidence': round(result.get('confidence', 0.85), 3),
                'annual_energy_mwh': round(result['predicted_energy_kwh'] / 1000, 2),
                'daily_energy_kwh': round(result['predicted_energy_kwh'] / 365, 2)
            },
            'metrics': result['input_data']['metrics']
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/compare', methods=['POST'])
def compare():
    try:
        if predictor is None:
            return jsonify({'success': False, 'error': 'Energy predictor not initialized'}), 500
            
        data = request.get_json()
        area = float(data.get('area', 100))
        
        results = {}
        
        if data.get('location_type') == 'coordinates':
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            location_info = f"({lat:.4f}, {lon:.4f})"
            for energy_type in ['solar', 'wind', 'hydro']:
                result = predictor.predict_energy(lat, lon, energy_type, area)
                results[energy_type] = {
                    'energy_kwh': round(result['predicted_energy_kwh'], 2),
                    'suitability': round(result['suitability_score'], 3),
                    'confidence': round(result.get('confidence', 0.85), 3)
                }
        else:
            city_name = data['city_name']
            location_info = city_name
            for energy_type in ['solar', 'wind', 'hydro']:
                result = predictor.predict_energy_city(city_name, energy_type, area)
                results[energy_type] = {
                    'energy_kwh': round(result['predicted_energy_kwh'], 2),
                    'suitability': round(result['suitability_score'], 3),
                    'confidence': round(result.get('confidence', 0.85), 3)
                }
        
        return jsonify({
            'success': True,
            'location': location_info,
            'area': area,
            'comparison': results
        })
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/engineers', methods=['POST'])
def add_engineer():
    try:
        data = request.get_json()
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO engineers (name, email, specialization, experience_years, location, hourly_rate, rating, bio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['name'], data['email'], data['specialization'],
            data.get('experience_years', 0), data.get('location', ''),
            data.get('hourly_rate', 0), data.get('rating', 0), data.get('bio', '')
        ))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Engineer added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/marketplace', methods=['POST'])
def add_marketplace_item():
    try:
        data = request.get_json()
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO marketplace (seller_name, seller_email, item_type, item_name, description, price, quantity, location, condition, contact_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['seller_name'], data['seller_email'], data['item_type'],
            data['item_name'], data.get('description', ''), 
            data.get('price', 0), data.get('quantity', 1),
            data.get('location', ''), data.get('condition', 'new'),
            data.get('contact_method', 'email')
        ))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Item listed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/energy-offers', methods=['POST'])
def add_energy_offer():
    try:
        data = request.get_json()
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO energy_offers (seller_name, seller_email, energy_type, capacity_kwh, price_per_kwh, location, availability_date, duration_months, contact_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['seller_name'], data['seller_email'], data['energy_type'],
            data['capacity_kwh'], data.get('price_per_kwh', 0),
            data.get('location', ''), data.get('availability_date', ''),
            data.get('duration_months', 12), data.get('contact_method', 'email')
        ))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Energy offer listed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Print debug information
    print("=" * 50)
    print("ENERGY PLATFORM STARTING...")
    print("=" * 50)
    print(f"Current directory: {os.getcwd()}")
    print(f"Templates folder exists: {os.path.exists('templates')}")
    print(f"Static folder exists: {os.path.exists('static')}")
    
    if os.path.exists('templates'):
        print(f"Files in templates: {os.listdir('templates')}")
    
    missing_templates = check_templates()
    if missing_templates:
        print(f"  MISSING TEMPLATES: {missing_templates}")
    else:
        print("All templates found!")
    
    print("=" * 50)
    print("Starting server at http://localhost:5000")
    print("Visit /debug for detailed status")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)