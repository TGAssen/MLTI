from sklearn.datasets import load_iris
from random import seed,randint
import numpy as np

class Perceptron(object):
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
        #Activation check
        if dotproduct+self.bias >= 0:
            self.output = 1
        return self.output
    

    def __str__(self):
        return "Perceptron with weights: "+ str(self.weights) + " and bias: " + str(self.bias)

class PerceptronLayer(object):
    def __init__(self, perceptrons):
        self.perceptrons=perceptrons

    def __str__(self):
        for i in self.perceptrons:
            print(i)

    def generateOutput(self,input):
        output=[]
        for j in self.perceptrons:
            output.append(j.act_fn(input))
        return output

class PerceptronNetwork(object):
    def __init__(self,layers):
        self.layers = layers
    
    def __str__(self):
        print("start perceptron network: ")
        print(self.layers[-1])
        return "end of network"

    def feedForward(self,input):
        for i in self.layers:
            input = i.generateOutput(input)
        return input

def train(epochs,p,testinput,testresult):
    while epochs > 0:

        for i,j in enumerate(testinput):
            p.act_fn(j)
            p.update(testresult[i],j)
        p.loss()

        epochs-=1


andtestinput = [[1,1],[1,0],[0,1],[0,0]]
andresult = [1,0,0,0]
pand = Perceptron([-0.5,0.5],-1.5)
#XOR perceptron zal niet werken, omdat er geen één lijn te vinden is welke de classificaties scheid. 
xortestinput = [[1,1],[1,0],[0,1],[0,0]]
xorresult = [0,1,1,0]
pxor = Perceptron()

seed(1667889)

iris = load_iris()
st_and_vc =iris.data[0:100]
st_and_vc_t = iris.target[0:100]
vc_and_vg = iris.data[50:150]
vc_and_vg_t = iris.target[50:150]
#normalize target
for i in range(len(vc_and_vg_t)):
    vc_and_vg_t[i] -=1
#print(i_data)
#print(i_target)


vcvg_irisperceptron= Perceptron(bias=randint(-2,2))
stvc_irisperceptron= Perceptron(bias=randint(-2,2))

train(100,pand,andtestinput,andresult)
print("3A: And:")
print(pand)

train(100,pxor,xortestinput,xorresult)
print("3B: Xor:")
print(pxor)

train(100,stvc_irisperceptron,st_and_vc,st_and_vc_t)
print("3CI:")
print(stvc_irisperceptron)

train(100,vcvg_irisperceptron,vc_and_vg,vc_and_vg_t)
print("3CII:")
print(vcvg_irisperceptron)





