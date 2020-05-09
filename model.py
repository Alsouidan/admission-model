import pandas as pd
import numpy as np


def initialize_parameters_log(layer_dims):
    parameters = {}
    # W=np.random.randn(1,X.shape[0])*0.1
    for i in range(1, len(layer_dims)):
        # print(layer_dims[i-1])
        # print(layer_dims[i])
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.05
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters


def forward_prop_log(A_prev, W, b, activation):
    Z = np.dot(W, A_prev)+b
    if activation == 'sigmoid':
        A = 1/(1+np.exp(-1*Z))
    else:
        A = np.maximum(Z, 0)
    cache = {'A_prev': A_prev, 'W': W, 'b': b, 'Z': Z}
    return A, cache


def relu_back(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid_back(Z):
    return Z*(1-Z)


def forward_prop_deep(X, parameters, layers):
    caches = []
    prev_A = X
    A=X
    for i in range(1,layers-1):
        prev_A = A
        A, cache = forward_prop_log(prev_A, parameters['W'+str(i)],parameters['b'+str(i)], 'relu')
        # print(parameters['W'+str(i)].shape)
        # print("Previous A shape is :"+str(prev_A.shape))
        # print("A shape is :"+str(A.shape))
        cache['prev_A'] = prev_A
        caches.append(cache)
    AL, last_cache = forward_prop_log(
        A, parameters['W'+str(layers-1)],parameters['b'+str(layers-1)], 'sigmoid')
    # print(parameters['W'+str(layers-1)].shape)
    # print("Previous A shape is :"+str(A.shape))
    # print("AL shape is :"+str(AL.shape))
    last_cache['prev_A'] = A
    caches.append(last_cache)
    return AL, caches


def backward_prop_log(dA, Z, prev_A, W, activation, m):
    # print('New Back Prop')
    dZ = 0
    if activation == 'sigmoid':
        dZ = dA*sigmoid_back(Z)
    else:
        dZ = dA*relu_back(Z)
    # print('dZ shape '+str(dZ.shape))
    # print('Z shape '+str(Z.shape))
    # print('previous A is:'+str(prev_A.shape))
    # print('dA shape is :'+str(dA.shape))
    dW = 1/m*np.dot(dZ, prev_A.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    grads = {'dW': dW, 'dZ': dZ, 'db' : db, 'dA_prev': dA_prev}
    return grads


def model(X, Y, iterations, learning_rate, layer_dims):
    # print(len(layer_dims))
    grads = {}
    parameters = initialize_parameters_log(layer_dims)
    m = X.shape[1]
    L = len(layer_dims)
    for j in range(iterations):
        AL, caches = forward_prop_deep(X, parameters, len(layer_dims))
        dAL = -1*(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads_temp = backward_prop_log(dAL, caches[len(caches)-1]['Z'], caches[len(
            caches)-1]['prev_A'], caches[len(caches)-1]['W'], 'sigmoid', m)
        parameters['W'+str(L-1)] = parameters['W'+str(L-1)] - \
            learning_rate*grads_temp['dW']
        parameters['b'+str(L-1)] = parameters['b'+str(L-1)] - \
            learning_rate*grads_temp['db']
        # print("HI")
        for i in reversed(range(1,L-1)):
            # print('hi')
            grads_temp = backward_prop_log(
                grads_temp['dA_prev'], caches[i-1]['Z'], caches[i-1]['prev_A'], caches[i-1]['W'], 'relu',m)
            parameters['W'+str(i)]=parameters['W'+str(i)]-learning_rate*grads_temp['dW']
            parameters['b'+str(i)]=parameters['b'+str(i)]-learning_rate*grads_temp['db']
        print(compute_cost_log(AL,Y,X))
    return parameters

def compute_cost_log(A, Y, X):
    J = (-1/X.shape[1])*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))
    J = np.squeeze(J)
    return J


def update_parameters(dW, db, W, b, learning_rate):
    new_W = W-learning_rate*dW
    new_b = b-learning_rate*db
    return new_W, new_b
# def backward_prop_log(A,Y,X):
#     dW= (1/X.shape[1])*(np.dot(X, (A-Y).T))
#     db = (1/X.shape[1])*np.sum(A-Y)
#     return dW,db
def predict(X,parameters,layers):
    AL, caches = forward_prop_deep(X, parameters, layers)
    Y_prediction = np.zeros((1, X.shape[1]))
    for i in range(AL.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        # START CODE HERE ### (≈ 4 lines of code)
        if(AL[0, i] > 0.5):
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
        ### END CODE HERE ###
    return Y_prediction


data = pd.read_csv("Admission_Predict.csv")
features = ['GRE Score', 'TOEFL Score',
            'University Rating', 'SOP', 'LOR ', 'CGPA']
feat = {}
for i in range(len(features)):
    feat[features[i]] = data[features[i]].values - \
        (sum(data[features[i]])/len(data[features[i]]))/(sum(np.square(data[features[i]]))/len(data[features[i]]))
Y_values = data["Chance of Admit "].values
X = []
Y = []
for i in range(int(len(data)*0.70)):
    features_x = []
    for j in features:
        features_x.append([feat[j][i]])
    X.append(features_x)
    if(Y_values[i] >= 0.5):
        Y.append([1])
    else:
        Y.append([0])
X_train = np.array(X).reshape((6, int(len(data)*0.70)))
Y_train = np.array(Y).T
X = []
Y = []
i = X_train.shape[1]
while i < len(data):

    features_x = []
    for j in features:
        features_x.append([feat[j][i]])
    X.append(features_x)
    if(Y_values[i] >= 0.5):
        Y.append([1])
    else:
        Y.append([0])
    i += 1
X_test = np.array(X)
X_test = X_test.reshape((6, X_test.shape[0]))
Y_test = np.array(Y).T
params=model(X_train, Y_train, 1000, 0.05, [X_train.shape[0],6,6,5,4,3,3,1])

Y_prediction_test = predict(X_test,params,8)
Y_prediction_train = predict(X_train,params,8)
print("train accuracy: {} %".format(
    100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(
    100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
