import numpy as np
import matplotlib.pyplot as plt

class SNN :
    
    def __init__(self, layer_sizes, activations, learning_rate, max_iter):
        # This is the constructer of the model where it will initialize the hyperparameters
        #       of the model (layer_sizes, activations, learning_rate, max_iter) and initialize
        #       the paramaters to an empty diconnary and the costs to an empty list

        assert(len(layer_sizes) == len(activations) + 1)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.parameters = dict()
        self.costs=[]
        
    def fit(self, X, Y,verbose=False):
        # This methode assemble all the parts to build a L-Layer NN and find the optimal parametes
        #       for a specified input data X and output data Y
        
        # Arguments:
        # X -- data : size (n[0], m)
        # Y -- true "label" vector: size (1, m)
        # verbose -- if True, it prints the cost every 100 steps

        self.costs = []
        self.initialize()
        for i in range(self.max_iter):
            AL, caches = self.FullFW(X)
            self.costs.append(self.compute_cost(AL, Y))
            if verbose and i%100==0:
                print("Cost after iteration ",i," ",self.costs[-1])
            grads = self.FullBW(Y, AL, caches)
            self.update(grads)
            
    def initialize(self):
        # This method initializes the weights randomly and the bias vectors as a vector of zeros
        #       and store them in the attribute parameters

        # self.parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
        #                 Wl -- weight matrix of shape (layer_sizes[l], layer_sizes[l-1])
        #                 bl -- bias vector of shape (layer_sizes[l], 1)
        
        for i in range(1,len(self.layer_sizes)):
            self.parameters['W' + str(i)] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * 0.01
            self.parameters['b' + str(i)] = np.zeros((self.layer_sizes[i],1))
    
    def FwProp(self, A_prev, W, b, activ): 
        # This method implements a single forward propagation from layer l-1 to the layer l

        # Arguments:
        # A_prev -- A[l-1] activations from the previous layer : size (n[l-1], m)
        # W -- weights matrix : size (n[l], n[l-1])
        # b -- bias vector :  size (n[l], 1)
        # activation -- the activation to be used in this layer, stored as a string: "sigmoid" or "relu" or "tanh"

        # Returns:
        # A -- A[l] the output of the activation function size (n[l], m)
        # cache -- a tuple containing (Z, W, b, A_prev)
        #         this cache will be passed to the Backward propagation step

        Z = np.dot(W, A_prev) + b
        A = Z
        if activ == 'sigmoid':
            A = self.sigmoid(Z)   
        if activ == 'relu':
            A = self.relu(Z)
        if activ == 'tanh':
            A = self.tanh(Z)
        return A , (Z, W, b, A_prev)
       
    def FullFW(self, X):
        # This method implements the forward propagation through all the network
        
        # Arguments:
        # X -- input data : size (n[0], m)
        
        # Returns:
        # AL -- last activation value : size (n[L],m)
        # caches -- list of caches containing every cache returned from FwProp method [(Z1, W1, b1, A[0]),...,(ZL, WL, bL, A[L])]

        caches = []
        A = X
        for i in range(1,len(self.activations)+1):
            Wi, bi = self.parameters['W' + str(i)], self.parameters['b' + str(i)]
            A, cache = self.FwProp(A, Wi, bi, self.activations[i-1])
            caches.append(cache)
        return A, caches
    
    def BwProp(self, dAl, cache, activ): 
        # This method implements a single backward propagation from layer l to the layer l-1
        
        # Arguments:
        # dAl -- post-activation gradient for current layer l 
        # cache -- tuple of values (Zl, Wl, bl, A[l-1]) comming from FwProp method for layer l
        # activation -- the activation to be used in this layer, stored as a string: "sigmoid" or "relu" or "tanh"
        
        # Returns:
        # dA_prev -- dA[l-1] Gradient of the cost with respect to the activation of the previous layer l-1 :same shape as A_prev
        # (dWl, dbl) -- a tuple of gradients of Wl and bl 
        #        dWl -- Gradient of the cost with respect to Wl : same shape as Wl
        #        dbl -- Gradient of the cost with respect to bl : same shape as bl
    
        Zl, W, b, A_prev = cache
        m = A_prev.shape[1]
        dZl = dAl
        if activ == 'sigmoid':
            dZl = dAl * self.back_sigmoid(Zl) #back_sigmoid return σ'(Z)
        if activ == 'relu':
            dZl = dAl * self.back_relu(Zl) #back_sigmoid return relu'(Z)
        if activ == 'tanh':
            dZl = dAl * self.back_tanh(Zl) #back_sigmoid return tanh'(Z)
        dWl = np.dot(dZl, A_prev.T)/m
        dbl = np.sum(dZl, axis=1, keepdims=True)/m
        dA_prev = np.dot(W.T,dZl)
        return dA_prev, (dWl, dbl)
    
    def FullBW(self, Y, AL, caches): 
        # This method implements the backward propagation through all the network
        
        # Arguments:
        # Y -- true "label" vector 
        # AL -- probability vector, output of the forward propagation 
        # caches -- list of caches containing every cache returned during the forward propagation
        
        # Returns:
        # grads -- A dictionary with the gradients
        #         grads["dW" + str(l)] = ...
        #         grads["db" + str(l)] = ... 
        grads = dict()
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA_prev, grad= self.BwProp(dAL, caches[-1], self.activations[-1])
        grads['dW' + str(len(self.activations))], grads['db' + str(len(self.activations))] = grad
        for i in reversed(range(len(self.activations)-1)):
            dA_prev, grad= self.BwProp(dA_prev, caches[i], self.activations[i])
            grads['dW' + str(i+1)], grads['db' + str(i+1)] = grad
        return grads
    
    def update(self, grads):
        # This method updates the parameters using gradient descent 
    
        # Arguments:
        # grads -- python dictionary containing all parameters gradients
        
        # Update :
        #       W -- W = W - α x dW
        #       b -- b = b - α x db

        for i in range(len(self.parameters)//2):
            self.parameters['W' + str(i+1)] -= self.learning_rate * grads['dW' + str(i+1)]
            self.parameters['b' + str(i+1)] -= self.learning_rate * grads['db' + str(i+1)]
    
    def predict(self, X, binary=False):
        # This method is used to predict the results of a L-layer neural network.
        
        # Arguments:
        # X -- data set of examples you would like to label
        
        # Returns:
        # if binary == False : p -- probability predictions for the given dataset X
        # if binary == True : p --  predictions for the given dataset X
        probas = self.FullFW(X)[0]
        if binary :
            m = X.shape[1]
            p = np.zeros((1,m))

            # convert probas to 0/1 predictions
            for i in range(0, m):
                if probas[0,i] > 0.5:
                    p[0,i] = 1
                else:
                    p[0,i] = 0
            return p
        else:
            return probas
    def accuracy(self, Y_true, Y_pred):
        #This method is used to calculate the accuracy of the model predicton

        # Arguments:
        # Y_true -- true labels
        # Y_pred -- predicted labels

        # Returns:
        # accuracy -- number of correct answers divided by the total exemples
        assert(Y_true.shape == Y_pred.shape)
        m = Y_true.shape[1]
        return np.sum((Y_true == Y_pred)/m)

    def compute_cost(self, AL, Y): 
        
        # This method implement the cost function defined by (1/m)Σ(- y log(y_hat) - (1-y) log(1-y_hat)) (cross-entropy cost)

        # Arguments:
        # AL -- probability vector corresponding to your label predictions : shape (1, m)
        # Y -- true "label" vector : shape (1, m)

        # Returns:
        # cost -- cross-entropy cost
        
        m = Y.shape[1]
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        return cost
    
    def plot_cost(self):
        # This method is used to plot the cost function after each iteration 
        #       using the data stored in self.costs      
        plt.plot(np.linspace(1,self.max_iter,self.max_iter),self.costs)
        plt.title("learning rate : "+str(self.learning_rate))
        plt.xlabel("#iteration")
        plt.ylabel("Cost functon")
        plt.show()

    def sigmoid(self, Z):
        # Implements the sigmoid activation in numpy
        
        # Arguments:
        # Z -- numpy array of any shape
        
        # Returns:
        # A -- output of sigmoid(z), same shape as Z

        A = 1/(1+np.exp(-Z))
        return A

    def back_sigmoid(self, Z):
        # Implements the derivative of sigmoid activation in numpy
        
        # Arguments:
        # Z -- numpy array of any shape
        
        # Returns:
        # A -- output of  sigmoid'(z), same shape as Z
        A = 1/(1+np.exp(-Z))
        return A * (1 - A)

    def relu(self, Z):
        # Implements the relu activation in numpy
        
        # Arguments:
        # Z -- numpy array of any shape
        
        # Returns:
        # A -- output of relu(z), same shape as Z
        A = np.maximum(0,Z)
        return A

    def back_relu(self, Z):
        # Implements the derivative of relu activation in numpy
        
        # Arguments:
        # Z -- numpy array of any shape
        
        # Returns:
        # A -- output of  relu'(z), same shape as Z
        Z = np.array(Z)
        A=np.zeros(Z.shape,dtype='int64')
        A[Z>0] = 1
        return A

    def tanh(self, Z):
        # Implements the tanh activation in numpy
        
        # Arguments:
        # Z -- numpy array of any shape
        
        # Returns:
        # A -- output of tanh(z), same shape as Z
        return np.divide(1-np.exp(-2*Z),1+np.exp(-2*Z))

    def back_tanh(self, Z):
        # Implements the derivative of tanh activation in numpy
        
        # Arguments:
        # Z -- numpy array of any shape
        
        # Returns:
        # A -- output of  tanh'(z), same shape as Z
        A = (1-np.exp(-2*Z))/(1+np.exp(-2*Z))
        return 1 - np.power(A,2)
