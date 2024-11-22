from flask import Flask, render_template, request, jsonify
from main import qry  # Import the qry function from main.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/query', methods=['POST'])
def query():
    user_input = request.form["query"]
    if not user_input:
        return jsonify({"error": "No query provided"}), 400
    try:
        response = qry(user_input)  # Call the qry function from main.py
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
