import numpy as np
from flask import Flask, request, jsonify, render_template
from tickle import model

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
model = model("18-8-2024_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_id = int(request.form['user_id'])  # Extract user ID from form input
    recommendations = model.recommend(user_id, 10)  # Use your model's recommendation method
    
    # Assuming recommendations is a list of movie serial numbers
    # Made by Naitik Dobariya
    recommendation_text = ', '.join([str(movie) + '\n' for movie in recommendations])

    return render_template('index.html', prediction_text=f'Recommended movies (serial numbers) \n {recommendation_text}')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    user_id = data['user_id']
    recommendations = model.recommend(user_id)

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
