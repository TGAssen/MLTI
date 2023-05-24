class Perceptron(object):
    def __init__(self, weights=[],bias = -1,output = 0):
        self.weights=weights
        self.bias = bias
        self.output = output
        
    def update(self,target,input):
        error = target - self.output
        n=0.1
        for i,j in enumerate(self.weights):
            deltaweight = n*error*input[i]
            self.weights[i]=deltaweight+j
        deltabias = n*error
        #self.bias += deltabias

        
    def loss(self):
        pass

    def act_fn(self,input):
        dotproduct = int(0)
        temp = len(input) - len(self.weights)
        while temp > 0 :
            self.weights.append(0)
            temp -= 1

        
        for i,j in enumerate(input):
            dotproduct = int(j) * self.weights[i] + dotproduct

        if dotproduct+self.bias > 0:
            self.output = 1
        else:
            self.output=0
        return self.output
    

    def __str__(self):
        return "Perceptron with weights: "+ str(self.weights) + " and bias: " + str(self.bias) + " With output: " + str(self.output)

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
    
andtestinput = [[1,1],[1,0],[0,1],[0,0]]
andresult = [1,0,0,0]
epochs = 0
p = Perceptron()
while epochs < 10:
    for i,j in enumerate(andtestinput):
        p.act_fn(j)
        p.update(andresult[i],j)
        print(p)
    epochs+=1

