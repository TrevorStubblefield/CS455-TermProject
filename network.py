
import numpy as np
import scg as scg
from copy import copy

class NetworkModeler:

    def __init__(self, nInputAttributes, hiddenLayersSpec, numOutputs):        
        try:
            inputAndHiddenLayersSpecList = [nInputAttributes] + list(hiddenLayersSpec)
            hiddenLayersSpecList = list(hiddenLayersSpec)
        except:
            inputAndHiddenLayersSpecList = [nInputAttributes] + [hiddenLayersSpec]
            hiddenLayersSpecList = [hiddenLayersSpec]
            
        self.Vs = [] # list of matrices of hidden layer weights
        
        for i in range(len(inputAndHiddenLayersSpecList)-1):
            sqrtOfCols = np.sqrt(inputAndHiddenLayersSpecList[i])
            V=1/sqrtOfCols * np.random.uniform(-1, 1, size=(1+inputAndHiddenLayersSpecList[i], inputAndHiddenLayersSpecList[i+1]))
            self.Vs.append(V)
        
        # lone output layer weight matrix
        self.W = 1/np.sqrt(hiddenLayersSpecList[-1]) * np.random.uniform(-1, 1, size=(1+hiddenLayersSpecList[-1], numOutputs))
        
        self.nInputAttributes, self.hiddenLayersSpecList, self.numOutputs = nInputAttributes, hiddenLayersSpecList, numOutputs
        
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.trained = False
        self.reason = None
        self.errorTrace = None
        self.numberOfIterations = None

    def __repr__(self):
        str = 'Network({}, {}, {})'.format(self.nInputAttributes, self.hiddenLayersSpecList, self.numOutputs)
        # str += '  Standardization parameters' + (' not' if self.Xmeans == None else '') + ' calculated.'
        if self.trained:
            str += '\n   Network was trained for {} iterations. Final error is {}.'.format(self.numberOfIterations,
                                                                                           self.errorTrace[-1])
        else:
            str += '  Network is not trained.'
        return str
            
    def standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result
    
    def unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans
    
    def standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result
    
    def unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans
   
    def getSCGWtVectorFromWtMatrices(self, Vs, W):
        return np.hstack([V.flat for V in Vs] + [W.flat])

    def getWtMatricesFromSCGWtVector(self,w):
        first = 0
        nhs = self.hiddenLayersSpecList
        numInThisLayer = self.nInputAttributes # start with inputs
        for i in range(len(self.Vs)):
            wtVectorItem = w[first:first+(numInThisLayer+1)]
            self.Vs[i][:] =w[first:first+(numInThisLayer+1) * self.hiddenLayersSpecList[i]].reshape((numInThisLayer+1, 
                                                                                                self.hiddenLayersSpecList[i]))
            first += (numInThisLayer+1) * self.hiddenLayersSpecList[i]
            numInThisLayer = self.hiddenLayersSpecList[i]
        self.W[:] = w[first:].reshape((numInThisLayer+1, self.numOutputs))

     # takes in the input matrix, Output matrix
     # runs nIterations of scaled conjugate gradient descent
     # arrives at the best matrices of hidden layer weights and output layer weight matrix
     # at the end the best weights for each hidden layer and output layer are available in list of hidden layer weight matrices
     #likewise the best weights for output layer are available in output layer weight matrix
    def trainBySCG(self, X, T, nIterations=100, verbose=False, weightPrecision=0, errorPrecision=0, saveWeightsHistory=False):
        
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xconstant = self.Xstds == 0
            self.XstdsFixed = copy(self.Xstds)
            self.XstdsFixed[self.Xconstant] = 1
        X = self.standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1
        T = self.standardizeT(T)

        ## takes in flattened weight vector with minimized error function from previous backward pass
        ## returns the mse error function using neural network forward pass
        def errorFunctionOfWts(w):
            self.getWtMatricesFromSCGWtVector(w)
            Zprev = X
            for i in range(len(self.hiddenLayersSpecList)):
                V = self.Vs[i]
                # invoke hyperbolic tangent function in each hidden layer
                Zprev = np.tanh(Zprev @ V[1:,:] + V[0:1,:])  # handling bias weight without adding column of 1's
            Y = Zprev @ self.W[1:,:] + self.W[0:1,:]
            return np.mean((T-Y)**2)

        ## takes in flattened weight vector with minimized error function from previous backward pass
        ## runs descent and returns new flattened weight vector with minimized error function from this backward pass
        def gradientOfErrorFunctionOfWts(w):
            ## get new weights from last run of SCG
            self.getWtMatricesFromSCGWtVector(w)
            Zprev = X
            Z = [Zprev]
            for i in range(len(self.hiddenLayersSpecList)):
                V = self.Vs[i]
                Zprev = np.tanh(Zprev @ V[1:,:] + V[0:1,:])
                Z.append(Zprev)
            Y = Zprev @ self.W[1:,:] + self.W[0:1,:]
            delta = -(T - Y) / (X.shape[0] * T.shape[1])
            dW = 2 * np.vstack(( np.ones((1,delta.shape[0])) @ delta, 
                                 Z[-1].T @ delta ))
            dVs = []
            delta = (1 - Z[-1]**2) * (delta @ self.W[1:,:].T)
            for Zi in range(len(self.hiddenLayersSpecList), 0, -1):
                Vi = Zi - 1 # because X is first element of Z
                dV = 2 * np.vstack(( np.ones((1,delta.shape[0])) @ delta,
                                     Z[Zi-1].T @ delta ))
                dVs.insert(0,dV)
                delta = (delta @ self.Vs[Vi][1:,:].T) * (1 - Z[Zi-1]**2)
            # return the latest minimized error function weights packed as a flat vector
            return self.getSCGWtVectorFromWtMatrices(dVs, dW)

        scgresult = scg.scg(self.getSCGWtVectorFromWtMatrices(self.Vs, self.W),
                            errorFunctionOfWts, gradientOfErrorFunctionOfWts,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            verbose=verbose,
                            ftracep=True,
                            xtracep=saveWeightsHistory)

        self.getWtMatricesFromSCGWtVector(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = np.sqrt(scgresult['ftrace']) # * self.Tstds # to unstandardize the MSEs
        self.numberOfIterations = len(self.errorTrace)
        self.trained = True
        self.weightsHistory = scgresult['xtrace'] if saveWeightsHistory else None
        return self

    def predict(self, X, allOutputs=False):
        Zprev = self.standardizeX(X)
        Z = [Zprev]
        for i in range(len(self.hiddenLayersSpecList)):
            V = self.Vs[i]
            Zprev = np.tanh( Zprev @ V[1:,:] + V[0:1,:])
            Z.append(Zprev)
        Y = Zprev @ self.W[1:,:] + self.W[0:1,:]
        Y = self.unstandardizeT(Y)
        return (Y, Z[1:]) if allOutputs else Y

    def getNumberOfIterations(self):
        return self.numberOfIterations
    
    def getErrors(self):
        return self.errorTrace
        
    def getWeightsHistory(self):
        return self.weightsHistory
