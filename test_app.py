from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Hello World!</h1>
        <p>If you can see this, Flask is working!</p>
        <a href="/energy">Go to Energy Page</a>
    </body>
    </html>
    """

@app.route('/energy')
def energy():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Energy Page</title>
    </head>
    <body>
        <h1>Energy Page</h1>
        <p>This is a simple energy page.</p>
        <a href="/">Back to Home</a>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting test server...")
    app.run(debug=True, host='0.0.0.0', port=5000)