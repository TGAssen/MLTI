from sklearn.datasets import load_iris
from random import seed,randint
import numpy as np
import math

class Neuron(object):
    def __init__(self,name,bias =float(0)):
        self.weights = []
        self.bias = bias
        self.output = 0
        self.error = float(0)
        self.input = []
        self.w_x_wdelta= []
        self.n = float(1)     #correction step
        self.name=name
    
    """ Als je een default lijst meegeeft in de initiatie parameters, word deze gedeeld tussen alle objecten die geen lijst meekrijgen.
    Zodoende worden de weights dan door alle neuronen gebruikt tenzij vooraf gedefinieerd. 
    Dus heb ik er voor gekozen dat het niet mogelijk is weights te initieren vanuit de constructor """
    def setWeights(self,weights):
        self.weights=weights

    #calculating the error
    def calcError(self,prev_error,target=0):
        self.w_x_wdelta.clear()
        # Error calculation for a hidden layer
        if prev_error:
            sum_weight_deltas = 0 
            
            for i in range(len(prev_error)):
                sum_weight_deltas += prev_error[i]
            self.error = self.output*(1-self.output)*sum_weight_deltas
        # Error calculation for output layer
        else:
            self.error  = self.output*(1-self.output)*(-1*(target-self.output))
        #fill weightdelta list
        for i in self.weights:
            self.w_x_wdelta.append(i*self.error)
        #print(self.name," has error of: ",self.error)
        return self.w_x_wdelta
    #update function
    def update(self):
        t = self.error*self.n
        for i in range(len(self.weights)):
            tt = t*self.input[i]
       
            self.weights[i] -= tt
        self.bias -= t
        #print(self.name," has updated weights: " ,self.weights, " and bias: ", self.bias)
        return 0
    
    #activation function which also fills the neuron with weights on the first call.        
    def act_fn(self,input):
        self.input = input
        dotproduct = 0
        self.output = 0
        #Match the amount of weights to the amount of inputs.
        wlcheck = len(input) - len(self.weights)
        seed(1667889)
        #fill the weightlist with random weights
        while wlcheck > 0 :
            self.weights.append(randint(-2,2))
            if wlcheck == 1:
                self.bias = randint(-2,2)
            wlcheck -= 1
        #Calculate the dotproduct
        for i in range(len(input)):
            dotproduct += input[i] * self.weights[i]
        #Sigmoid activation function
        self.output = 1 -(1/(1+math.e**(dotproduct+self.bias)))
        #print(self.output)
        return self.output

    def __str__(self):
        return "Neuron "+str(self.name)+" with weights: "+ str(self.weights) + " and bias: " + str(self.bias)

class NeuronLayer(object):
    def __init__(self, neurons):
        self.neurons = neurons

    def __str__(self):
        for i in self.neurons:
            print(i)
        return "end of layer"
    #Method that activates each neurons activation function and returns a list with each output.
    def passOn(self,input):
        output=[]
        for j in self.neurons:
            output.append(j.act_fn(input))
        return output
    
    def calculateErrors(self,errorlist,target):
        next_errorlist = []
        if errorlist:
            #easy transpose to get the correct weightpairs to the corresponding neuron
            np_errorlist=np.asarray(errorlist)
            np_errorlist = np_errorlist.transpose()
            t_errorlist = np_errorlist.tolist()
            for i in range(len(self.neurons)):
                next_errorlist.append(self.neurons[i].calcError(t_errorlist[i])) 
        else:
            for i in range(len(self.neurons)):
                next_errorlist.append(self.neurons[i].calcError(errorlist,target[i]))
        return next_errorlist

    def calcLoss(self,errorlist):
        mse = 0
        for i in errorlist:
            mse+= i**2
        mse = mse/len(errorlist)*2
        return mse
 
    def updateNeurons(self):
        for i in self.neurons:
            i.update()
        return 0

class NeuronNetwork(object):
    def __init__(self,layers):
        self.layers = layers
    
    def __str__(self):
        print("start neuron network: ")
        for i in self.layers:
            print(i)
        return "end of network"

    def feedForward(self,input):
        for i in self.layers:
            input = i.passOn(input)
        return input

    def train(self,inputs,target,epochs=10):
        for p in range(epochs):
            # one iteration of all 
            for i in range(len(inputs)):
                self.feedForward(inputs[i])
                errorlist= []
                for j in reversed(self.layers):
                    errorlist = j.calculateErrors(errorlist,target[i])
                for j in reversed(self.layers):
                    j.updateNeurons()
            
            print((p+1),"/",epochs,"epochs ")
        return self





input1 = [[1],[0]]
input2 = [[1,1],[1,0],[0,1],[0,0]]
input2_target_ha = [[0,1],[1,0],[1,0],[0,0]]
input2_target_and = [[1],[0],[0],[0]]
input2_target_xor = [[0],[1],[1],[0]]

input3 = [[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]]

n1=NeuronLayer([Neuron("f"),Neuron("g"),Neuron("h")])
n2=NeuronLayer([Neuron("n2-1"),Neuron("n2-2"),Neuron("n2-3")])
n3=NeuronLayer([Neuron("s"),Neuron("c")])
n1.neurons[0].setWeights([0.0,0.1])
n1.neurons[1].setWeights([0.2,0.3])
n1.neurons[2].setWeights([0.4,0.5])
n3.neurons[0].setWeights([0.6,0.7,0.8])
n3.neurons[1].setWeights([0.9,1.0,1.1])
halfadder= NeuronNetwork([n1,n3])
andlayer=NeuronLayer([Neuron("and")])
andgate = NeuronNetwork([andlayer])
andgate.train(input2,input2_target_and,1000)
xorlayer1 = NeuronLayer([Neuron("andi1"),Neuron("andi2")])
xorlayer2 = NeuronLayer([Neuron("or")])
xorgate = NeuronNetwork([xorlayer1, xorlayer2])
xorgate.train(input2,input2_target_xor,1000)


halfadder.train(input2,input2_target_ha,1000)

print("output and with 1,1: ",andgate.feedForward([1,1]))
print("output and with 0,1: ",andgate.feedForward([1,0]))
print("output and with 1,0: ",andgate.feedForward([0,1]))
print("output and with 0,0: ",andgate.feedForward([0,0]))

print("output xor with 1,1: ",xorgate.feedForward([1,1]))
print("output xor with 0,1: ",xorgate.feedForward([1,0]))
print("output xor with 1,0: ",xorgate.feedForward([0,1]))
print("output xor with 0,0: ",xorgate.feedForward([0,0]))

print("output ha with 1,1: ",halfadder.feedForward([1,1]))
print("output ha with 0,1: ",halfadder.feedForward([0,1]))
print("output ha with 1,0: ",halfadder.feedForward([1,0]))
print("output ha with 0,0: ",halfadder.feedForward([0,0]))
#halfadder.train(input2,input2_target_ha,10000)
#print(halfadder)

#networkand.train(input2,input2_target_and,5000)
#print(networkand)
#print(networkand.feedForward([1,1]))
seed(1667889)

iris = load_iris()
iris_d =iris.data
iris_t = iris.target
irislayer1=NeuronLayer([Neuron("n1"),Neuron("n2"),Neuron("n3"),Neuron("n4")])
irislayer2=NeuronLayer([Neuron("o1"),Neuron("o2"),Neuron("o3")])
irisnetwork= NeuronNetwork([irislayer1,irislayer2])
irisnetwork.train(iris_d,iris_t,1000)
print(irisnetwork.feedForward(iris_d[0]))



