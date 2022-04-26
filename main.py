import pandas
import sklearn.metrics
import sklearn.neural_network

dataset = pandas.read_csv('predict_data.csv')
labelset = pandas.read_csv('out.csv')
X = dataset.values  # Data
Y = labelset.values[:,0].astype(int)  # Labels
print(Y)
# Split data set in train and test (use random state to get the same split every time, and stratify to keep balance)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=5,
                                                                            stratify=Y)
model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                             alpha=0.0001, batch_size='auto', learning_rate='constant',
                                             learning_rate_init=0.001, power_t=0.5,
                                             max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
                                             verbose=False, warm_start=False, momentum=0.9,
                                             nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                             beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                             n_iter_no_change=10)
model.fit(X_train, Y_train)
predictions = model.predict(X_train)
accuracy = sklearn.metrics.accuracy_score(Y_train, predictions)

predict_data = pandas.DataFrame(pandas.read_csv('predict_data.csv'))
# print(predict_data.values)
predictions = model.predict(predict_data)
output_list = predict_data.values
output = open('output.csv', 'w')
ind = 0
for li in output_list:
    o = list(li)
    o.append(predictions[ind])
    output.write(str(o)[1:-1] + '\n')
    ind += 1
output.close()
