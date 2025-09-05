Manual Linear Regression

def forward_linear_regression(X_batch: ndarray,
                              y_batch: ndarray,
                              weights: Dict[str, ndarray]
                              )-> Tuple[float, Dict[str, ndarray]]:
    '''
    Forward pass for the step-by-step linear regression.
    '''
    # assert batch sizes of X and y are equal
    assert X_batch.shape[0] == y_batch.shape[0]

    # assert that matrix multiplication can work
    assert X_batch.shape[1] == weights['W'].shape[0]

    # assert that B is simply a 1x1 ndarray
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    # compute the operations on the forward pass
    N = np.dot(X_batch, weights['W'])

    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    # save the information computed on the forward pass
    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info

def to_2d_np(a: ndarray, 
             type: str = "col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)

def permute_data(X: ndarray, y: ndarray):
    '''
    Permute X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

def loss_gradients(forward_info: Dict[str, ndarray],
                   weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    '''
    Compute dLdW and dLdB for the step-by-step linear regression model.
    '''
    batch_size = forward_info['X'].shape[0]

    dLdP = -2 * (forward_info['y'] - forward_info['P'])

    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'], (1, 0))
    
    # need to use matrix multiplication here,
    # with dNdW on the left (see note at the end of last chapter)    
    dLdW = np.dot(dNdW, dLdN)

    # need to sum along dimension representing the batch size:
    # see note near the end of the chapter    
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB

    return loss_gradients

forward_info, loss = forward_loss(X_batch, y_batch, weights)
    loss_grads = loss_gradients(forward_info, weights)
for key in weights.keys(): # 'weights' and 'loss_grads' have the same keys weights[key] -= learning_rate * loss_grads[key]

# Code below runs the train function for certain number of cycles through the entire training set
train_info = train(X_train, y_train,
                       learning_rate = 0.001,
                       batch_size=23,
                       return_weights=True,
                       seed=80718)

def predict(X: ndarray,
            weights: Dict[str, ndarray]):
    '''
    Generate predictions from the step-by-step linear regression model.
    '''

    N = np.dot(X, weights['W'])

    return N + weights['B']

    preds = predict(X_test, weights) # weights = train_info[0]

def mae(preds: ndarray, actuals: ndarray): '''
Compute mean absolute error.
'''
return np.mean(np.abs(preds - actuals))

def rmse(preds: ndarray, actuals: ndarray): '''
Compute root mean squared error.
'''
return np.sqrt(np.mean(np.power(preds - actuals, 2)))

def forward_loss(X: ndarray, y: ndarray,
                     weights: Dict[str, ndarray]
                     ) -> Tuple[Dict[str, ndarray], float]:
'''
Compute the forward pass and the loss for the step-by-step neural network model.
'''
M1 = np.dot(X, weights['W1'])
        N1 = M1 + weights['B1']
        O1 = sigmoid(N1)
        M2 = np.dot(O1, weights['W2'])
        P = M2 + weights['B2']
        loss = np.mean(np.power(y - P, 2))
        forward_info: Dict[str, ndarray] = {}
        forward_info['X'] = X
        forward_info['M1'] = M1
        forward_info['N1'] = N1
        forward_info['O1'] = O1
        forward_info['M2'] = M2
        forward_info['P'] = P
        forward_info['y'] = y
return forward_info, loss

forward_info, loss = forward_loss(X_batch, y_batch, weights)
    loss_grads = loss_gradients(forward_info, weights)
for key in weights.keys():
weights[key] -= learning_rate * loss_grads[key]

preds = predict(X_test, weights)

def predict(X: ndarray,
weights: Dict[str, ndarray]) -> ndarray:
'''
Generate predictions from the step-by-step neural network model. '''
M1 = np.dot(X, weights['W1'])
        N1 = M1 + weights['B1']
        O1 = sigmoid(N1)
        M2 = np.dot(O1, weights['W2'])
        P = M2 + weights['B2']
return P













