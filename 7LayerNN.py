from numpy import exp, array, random, dot, float64, nditer

class NeuronLayer():
    def __init__(self, numNeurons, numInputs):
        self.weights = 2 * random.random((numInputs, numNeurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoidDerivative(self, x):
        return x * (1 - x)

    def train(self, trainingSetInputs, trainingSetOuputs, numIterations):
        for i in range (0, numIterations):
            layer1Output, layer2Output = self.think(trainingSetInputs)

            layer2Error = trainingSetOutputs - layer2Output
            layer2Delta = layer2Error * self.__sigmoidDerivative(layer2Output)

            layer1Error = layer2Delta.dot(self.layer2.weights.T)
            layer1Delta = layer1Error * self.__sigmoidDerivative(layer1Output)
            
            layer1Adjustment = trainingSetInputs.T.dot(layer1Delta)
            layer2Adjustment = layer1Output.T.dot(layer2Delta)

            self.layer1.weights += layer1Adjustment
            self.layer2.weights += layer2Adjustment

    def think(self, inputs):
        layer1Out = self.__sigmoid(dot(inputs, self.layer1.weights))
        layer2Out = self.__sigmoid(dot(layer1Out, self.layer2.weights))
        return layer1Out, layer2Out

def findHighest(output):
    best = 0
    i = 0
    index = 0
    for num in nditer(output):
        if (num > best):
            best = num
            index = i
        i += 1
    return index

if __name__ == "__main__":
    #Creating the random starting weights
    #random.seed(1)
    layer1 = NeuronLayer(8, 7)
    layer2 = NeuronLayer(10, 8)
    network = NeuralNetwork(layer1, layer2)

    #Defining the training set
    trainingSetInputs = array([[1,1,1,1,1,1,0],[0,1,1,0,0,0,0],[1,1,0,1,1,0,1],
                               [1,1,1,1,0,0,1],[0,1,1,0,0,1,1],[1,0,1,1,0,1,1],
                               [1,0,1,1,1,1,1],[1,1,1,0,0,0,0],[1,1,1,1,1,1,1],
                               [1,1,1,1,0,1,1]])
    trainingSetOutputs = array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])
    #Training the network using the training set 60 000 times
    ITERATIONS = 60000
    network.train(trainingSetInputs, trainingSetOutputs, ITERATIONS)

    file = open("in.txt", 'r')
    outFile = open("out.txt", 'w')
    for line in file:
        if (line[0] == '['):
            temp = []
            for char in line.rstrip("\n\r"):
                if (not char == '[' and not char == ',' and not char == ']'):
                    temp.append(char)
            newInput = array(temp, dtype=float64)
            hiddenState, output = network.think(newInput)
            outFile.write('[')
            i = findHighest(output)
            outFile.write(str(i))
            outFile.write("]\n\r\n\r")
    outFile.close()
