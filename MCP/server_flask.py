
"""

HTTP Web Service with Multiply Function

Installation:
pip install flask

Usage:
python server.py

The server will run on http://0.0.0.0:5000
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@app.route('/multiply', methods=['POST'])
def multiply_endpoint():
    """
    Multiply two integers via HTTP POST
    
    Request body (JSON):
    {
        "a": 5,
        "b": 3
    }
    
    Response:
    {
        "result": 15,
        "calculation": "5 × 3 = 15"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        a = data.get('a')
        b = data.get('b')
        
        if a is None or b is None:
            return jsonify({"error": "Both 'a' and 'b' parameters are required"}), 400
        
        if not isinstance(a, int) or not isinstance(b, int):
            return jsonify({"error": "Both 'a' and 'b' must be integers"}), 400
        
        result = multiply(a, b)
        
        return jsonify({
            "result": result,
            "calculation": f"{a} × {b} = {result}"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "multiply-server"}), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "service": "Multiply Web Service",
        "endpoints": {
            "/multiply": {
                "method": "POST",
                "description": "Multiply two integers",
                "example": {
                    "request": {"a": 5, "b": 3},
                    "response": {"result": 15, "calculation": "5 × 3 = 15"}
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check"
            }
        }
    }), 200

if __name__ == '__main__':
    print("Starting web service on http://0.0.0.0:5000")
    print("API endpoint: POST http://0.0.0.0:5000/multiply")
    app.run(host='0.0.0.0', port=5000, debug=True)


"""
curl -X POST http://localhost:5000/multiply \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}'

"""