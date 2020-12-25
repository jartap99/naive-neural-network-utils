#
# @rajeevp
#

import numpy as np

class DNN(object):
    def __init__(self):
        np.random.seed(9076301)
        self.description = "DNN functions"
        self.seluScale = 1.05070098
        self.seluAlpha = 1.67326324

    def __str__(self):
        return self.description

    def conv2D(ifm, weights, biases, padding):
        """
        naive implementation of 2D convolution
        non - vectroized - intention is not to make it faster but to assess 
        funcitonal correctness; eventually helping in hardware acceleration libraries 
        """
        print("*** DNN.conv2D ***")
        print("ifm     : ", ifm.shape)
        print("weights : ", weights.shape)
        print("biases  : ", biases.shape)
        m, n, c = ifm.shape
        k, l, c1, numFilters = weights.shape
        if padding == "valid":
            I = m-k+1
            J = n-l+1
            ifmPadded = ifm
        else:
            I = m
            J = n
            ifmPadded = np.pad(ifm, (  ( (k-1)//2, (k-1)//2 ), ( (l-1)//2, (l-1)//2 ), (0,0) ), 
                                mode="constant")
        ofm = np.zeros((I, J, numFilters))
        print("ofm     : ", ofm.shape)

        for nc in range(numFilters):
            for i in range(I):
                for j in range(J):
                    patch = ifmPadded[i:i+k, j:j+l, :]
                    filt = weights[:,:,:,nc]
                    ofm[i,j,nc] = np.sum(np.multiply(patch, filt)) + biases[nc]
        return ofm
    
    def activation(self, ifm, activationType):
        """ supported types
        relu, selu, sigmoid, tanh, leakyrelu, swish
        non - vectroized - intention is not to make it faster but to assess 
        funcitonal correctness; eventually helping in hardware acceleration libraries 
        """
        def f_relu(x):
            if (x<0): return 0
            else: return x
        
        def f_leaky_relu(x):
            if (x<0): return 0.01 * x
            else: return x
        
        def f_selu(x):
            if (x>0): return self.seluScale * x
            else: return self.seluScale * self.seluAlpha * (np.exp(x) - 1)

        def f_sigmoid(x):
            return 1/(1+np.exp(-x))
        
        def f_tanh(x):
            return np.tanh(x)
        
        def f_swish(x):
            return x * f_sigmoid(x)

        funcDict = {"relu" : f_relu, 
                    "leaky_relu" : f_leaky_relu,
                    "selu" : f_selu,
                    "sigmoid" : f_sigmoid,
                    "tanh" : f_tanh,
                    "swish" : f_swish
                    }
        print("ifm.shape : ", ifm.shape)
        m, n, c = ifm.shape
        ofm = np.zeros(ifm.shape)
        
        for nc in range(c):
            for i in range(m):
                for j in range(n):
                    ofm[i,j,nc] = funcDict[activationType](ifm[i,j,nc])
        
        return ofm




if __name__ == "__main__":
    dnn = DNN()
    print(dnn)
    #ifm = np.random.rand(5,5,3)
    a, b = -1, 1
    ifm = (b - a) * np.random.random_sample((5,5,3)) + a
    print("inputs : ", ifm[:,0,0])
    
    for act in ["relu", "leaky_relu", "selu", "sigmoid", "tanh", "swish"]:
        ofm = dnn.activation(ifm, activationType=act) 
        print(act, "  : ", ofm[:,0,0])
    
