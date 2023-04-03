from operator import mod
import main
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

main.get_data()
main.create_model()
# model.plot_graphs()
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    print("data: ", data)
    
    # if data['question'] == 'traiN':
    #     model.create_model()
    #     return jsonify({'prediction': 'Model trained successfully'})
    # prediction = model.predict(data['question'])
    # return jsonify({'prediction': prediction})
    allergy = data['allergy']
    skin_concern = data['skin_concern']
    skin_sensitivity = data['skin_sensitivity']
    skin_tone = data['skin_tone']
    skin_type = data['skin_type']

    prediction = main.predict_product(allergy, skin_concern, skin_sensitivity, skin_tone, skin_type)
    return jsonify({'prediction': prediction})



