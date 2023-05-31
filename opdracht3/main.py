from sklearn.datasets import load_iris
from random import seed,randint
import numpy as np
import math



class Neuron(object):
    def __init__(self, weights=[],bias = float(0),output = 0):
        self.weights=weights
        self.bias = bias
        self.output = output
        self.errorlist=[]
        
    def update(self,target,input):
        error = target - self.output
        n=0.1 #correctionstep
        self.errorlist.append(error)
        for i,j in enumerate(self.weights):

            deltaweight = n*error*input[i]
            self.weights[i]=deltaweight+j
        deltabias = n*error
        self.bias += deltabias
    def getOutput(self):
        return self.output
        
    def loss(self):
        mse=0
        for i in self.errorlist:
            mse+=i*i/ len(self.errorlist)
        self.errorlist.clear()
        return mse

    def act_fn(self,input):
        dotproduct = int(0)
        self.output = 0
        #Match the amount of inputs to the amount of weights.
        temp = len(input) - len(self.weights)
        seed(1667889)
        while temp > 0 :
            self.weights.append(randint(-2,2))
            temp -= 1
        #Calculate the dotproduct
        for i,j in enumerate(input):
            dotproduct += int(j) * self.weights[i]
        self.output = 1/(1+math.e**(-1*dotproduct-self.bias))
        return self.output
    

    def __str__(self):
        return "Perceptron with weights: "+ str(self.weights) + " and bias: " + str(self.bias)

class NeuronLayer(object):
    def __init__(self, neurons):
        self.neurons=neurons

    def __str__(self):
        for i in self.neurons:
            print(i)

    def generateOutput(self,input):
        output=[]
        for j in self.neurons:
            output.append(j.act_fn(input))
        return output

class NeuronNetwork(object):
    def __init__(self,layers):
        self.layers = layers
    
    def __str__(self):
        print("start neuron network: ")
        print(self.layers[-1])
        return "end of network"

    def feedForward(self,input):
        for i in self.layers:
            input = i.generateOutput(input)
        return input

def train(epochs,n,testinput,testresult):
    while epochs > 0:

        for i,j in enumerate(testinput):
            n.act_fn(j)
            n.update(testresult[i],j)
        n.loss()

        epochs-=1
def run(n,input):
    for i in input:
        print(n.act_fn(i))


input1 = [1,0]
input2 = [[1,1],[1,0],[0,1],[0,0]]
input3 = [[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]]
neuronand = Neuron([1,1],1)
neuronnot = Neuron([-1],1)
neuronor = Neuron([2,2],-1)
neuronnor = Neuron([-1,-1,-1],1)
run(neuronand,input2)
run(neuronnot,input1)
run(neuronor,input2)
run(neuronnor,input3)


""" seed(1667889)

iris = load_iris()
st_and_vc =iris.data[0:100]
st_and_vc_t = iris.target[0:100]
vc_and_vg = iris.data[50:150]
vc_and_vg_t = iris.target[50:150]
#normalize target
for i in range(len(vc_and_vg_t)):
    vc_and_vg_t[i] -=1

vcvg_irisneuron= Neuron(bias=randint(-2,2))
stvc_irisneuron= Neuron(bias=randint(-2,2))

train(100,neuronand,andtestinput,andresult)
print("3A: And:")
print(neuronand)

train(100,neuronxor,xortestinput,xorresult)
print("3B: Xor:")
print(neuronxor)

train(100,stvc_irisneuron,st_and_vc,st_and_vc_t)
print("3CI:")
print(stvc_irisneuron)

train(100,vcvg_irisneuron,vc_and_vg,vc_and_vg_t)
print("3CII:")
print(vcvg_irisneuron)
 """




