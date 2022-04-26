import os, shutil
from flask_restful import Api, Resource
from flask import Flask, flash, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
import pandas
import sklearn.metrics
import sklearn.neural_network



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

UPLOAD_FOLDER = r'C:\Users\tk2an\Downloads\FinalProduct\FinalProduct\backend\uploads'
ALLOWED_EXTENSIONS = set(['csv', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class CustomNeuralNetwork():
    def __init__(self):
        self.MODEL = None

    def neural(self):
        dataset = pandas.read_csv(UPLOAD_FOLDER + r'\input.csv')
        labelset = pandas.read_csv(UPLOAD_FOLDER + r'\output.csv')
        X = dataset.values  # Data
        Y = labelset.values[:, 0].astype(int)  # Labels
        # Split data set in train and test (use random state to get the same split every time, and stratify to keep balance)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=5,
                                                                                    stratify=Y)
        self.MODEL = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                                          alpha=0.0001, batch_size='auto', learning_rate='constant',
                                                          learning_rate_init=0.001, power_t=0.5,
                                                          max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
                                                          verbose=False, warm_start=False, momentum=0.9,
                                                          nesterovs_momentum=True, early_stopping=False,
                                                          validation_fraction=0.1,
                                                          beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                          n_iter_no_change=10)
        self.MODEL.fit(X_train, Y_train)
        predictions = self.MODEL.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
        response = jsonify({'accuracy': accuracy})
        print(accuracy)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    def predict(self):
        target = os.path.join(UPLOAD_FOLDER)
        if not os.path.isdir(target):
            os.mkdir(target)
        file = request.files['predict_x']
        filename = 'predict.csv'
        destination = "/".join([target, filename])
        file.save(destination)

        predict_data = pandas.DataFrame(pandas.read_csv(UPLOAD_FOLDER + r'\predict.csv'))
        print(predict_data.values)
        predictions = self.MODEL.predict(predict_data)
        output_list = predict_data.values
        output = open(UPLOAD_FOLDER + '/prediction_final.csv', 'w')
        ind = 0
        for li in output_list:
            o = list(li)
            o.append(predictions[ind])
            output.write(str(o)[1:-1] + '\n')
            ind += 1
        output.close()
        response = jsonify({'predictions': output_list})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    
nn = CustomNeuralNetwork()

@app.route('/upload', methods=['POST'])
def fileUpload():
    target = os.path.join(UPLOAD_FOLDER)
    if not os.path.isdir(target):
        os.mkdir(target)
    print("welcome to upload")
    file = request.files['input']
    filename = 'input.csv'
    destination = "/".join([target, filename])
    file.save(destination)
    file2 = request.files['output']
    filename2 = 'output.csv'
    destination = "/".join([target, filename2])
    file2.save(destination)
    session['uploadFilePath'] = destination
    response = jsonify({'resp': 'uploaded file'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/neural', methods=['GET'])
def create_network():
    return nn.neural()


@app.route('/neural', methods=['POST'])
def neuralFunctPredict():
    nn.neural()
    print('entered neural function')
    return nn.predict()



if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True, host="0.0.0.0", port=1000, use_reloader=False)

flask_cors.CORS(app, expose_headers='Authorization')
