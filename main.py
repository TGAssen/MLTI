class Perceptron(object):
    def __init__(self,inputs,logicport="none", weights=[],bias = 0,output=0):
        logicport.casefold()
        self.logicport = logicport
        self.inputs = inputs
        match logicport:
            case "not":
                self.weights=[-1]
                self.bias = 1
            case "and": 
                self.weights = [1]
                self.bias = -1
            case "or": 
                self.weights = [2]
                self.bias = -1
            case "nor":
                self.weights = [-1]
                self.bias = 1
            case "nand":
                self.weights = [-1]
                self.bias = 2
            case "buffer":
                self.weights = [1]
                self.bias = 0
            case "none":
                self.weights = weights
                self.bias = bias
        self.output = output


   
    def act_fn(self,input):
        dotproduct = int(0)
        temp = len(input) - len(self.weights)
        while temp > 0 :
            self.weights.append(self.weights[0])
            temp -= 1

        
        for i,j in enumerate(input):
            dotproduct = int(j) * self.weights[i] + dotproduct

        if dotproduct+self.bias > 0:
            self.output = 1
        return self.output
    

    def __str__(self):
        return "Perceptron as " +  self.logicport + ". With output: " + str(self.output)

class PerceptronLayer(object):
    def __init__(self, perceptrons):
        self.perceptrons=perceptrons

    def __str__(self):
        print("start of layer: ")
        for i in self.perceptrons:
            print(i)
        return "end of layer"
    def generateoutput(self,input):

        output=[]
        for j in self.perceptrons:
            output.append(j.act_fn(input))
        return output

class PerceptronNetwork(object):
    def __init__(self,layers):
        self.layers = layers
    
    def __str__(self):
        print("start perceptron network: ")
        for i in self.layers:
            print(i)
        return "end of network"
    def feedforward(self,input):

        for i in self.layers:
            input = i.generateoutput(input)
        return input
    

twoinput=[1,1]
xor=PerceptronNetwork([PerceptronLayer([Perceptron(2,"or"),Perceptron(2,"nand")]),PerceptronLayer([Perceptron(2,"and")])])

halfadder=PerceptronNetwork([PerceptronLayer([Perceptron(2, s"none", [1,0]),Perceptron(2,"nand"),Perceptron(2,"none",[0,1])]),PerceptronLayer([Perceptron(3,"none",[-1,-1,0],2),Perceptron(3,"none",[0,-1,-1],2),Perceptron(3,"none",[0,1,0],0)]),PerceptronLayer([Perceptron(3,"none",[-1,-1,0],2),Perceptron(3,"none",[0,0,-1],1)])])

halfadder.feedforward([0,0])

print(halfadder)


