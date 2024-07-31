import math
import numpy as np
import pickle
from keras.datasets import mnist
from random import randrange
#https://yann.lecun.com/exdb/mnist/

(train_X, train_y), (test_X, test_y) = mnist.load_data()

global rng
rng = np.random.default_rng()


def Network_Function(x):
    return relu(x)
def Derivative_Network_Function(x):
    return derivative_relu(x)

def Neuron_Function(layer):
    global Network, Anti_Network
    Anti_Network[layer] = Network[layer]
    for Row in range(Rows):
        x = Network[layer,Row]
        Network[layer,Row] = Network_Function(x)
def Network_Types_Update():
    global Derivative_Network, Network, Anti_Outputs, Derivative_Outputs
    Derivative_Network = Network
    for Layer in range(Layers):
        for Row in range(Rows):
            x = Anti_Network[Layer,Row]
            Derivative_Network[Layer,Row] = Derivative_Network_Function(x)
    Anti_Outputs = []
    Derivative_Outputs = []
    for x in Outputs:
        x = math.log((1/x)-1)
        Anti_Outputs.append(x)
        Derivative_Outputs.append((math.exp(-x))/((1+math.exp(-x))**2))

def Cost(x,y):
    return (x - y) ** 2
def Function(x):
    y = []
    for i in range(10):
        if x == i:
            y.append(1)
        else:
            y.append(0)
    return y
def Average(x):
    try:
        return sum(x)/len(x)
    except:
        return x
def sigmoid(x):
    return 1/( 1 + math.exp(-x) )
def inv_sigmoid(x):
    return math.log((1/x)-1)
def derivative_sigmoid(x):
    return (math.exp(-x))/((1+math.exp(-x))**2)
def relu(x):
    return max(0,x)
def derivative_relu(x):
    if x > 0:
        return 1
    else:
        return 0

def Neural_Network_Run(x):
    global Network
    Network = np.zeros((Layers,Rows))
    #First Layer
    for Row in range(len(x)):
        Network[0] += x[Row] * Exterior_Weights[0][Row]
    Network[0] /= len(x)
    Network[0] += Biases[1][0]
    Neuron_Function(0)
    #Interior Layers
    for Layer in range(Layers - 1):
        Current_Network_IN = Network[Layer]
        Current_Weights = Interior_Weights[Layer]
        for Row in range(Rows):
            Network[Layer+1] += Current_Network_IN[Row] * Current_Weights[Row]
        Network[Layer+1] /= Rows
        Network[Layer+1] += Biases[1][Layer+1]
        Neuron_Function(Layer+1)
    #Last Layer
    y = np.zeros(Num_Of_Outputs)
    for Row in range(Rows):
        y += Network[Layers-1,Row] * Exterior_Weights[1][Row]
    y /= Rows
    y += Biases[2]
    for i in range(Num_Of_Outputs):
        y[i] = 1/(1 + math.exp(-y[i]))
    return y

def Gradient():
    global Num_Of_Inputs, Num_Of_Outputs, Interior_Weights, Exterior_Weights, Biases, Network, Correct_Outputs, Outputs, Inputs, Rows, Layers
    Gradient = [np.zeros((Layers - 1,Rows,Rows))]
    Gradient.append([np.zeros((Num_Of_Inputs,Rows)),np.zeros((Rows,Num_Of_Outputs))])
    Gradient.append([np.zeros(Num_Of_Inputs),np.zeros((Layers,Rows)),np.zeros(Num_Of_Outputs)])
    Network_Types_Update()
    Outputs_Derivative = []
    for Row1 in range(Num_Of_Outputs):
        y = (Outputs[Row1] - Correct_Outputs[Row1]) * Derivative_Outputs[Row1] * 2
        Outputs_Derivative.append(y)
        #Last Layer Biases
        Gradient[2][2][Row1] = y
        for Row2 in range(Rows):
            neuron = Network[Layers - 1, Row2]
            #Last Layer Weights
            Gradient[1][1][Row2,Row1] = y * neuron
    neuron_der = []
    for Row1 in range(Rows):
        neuron_der.append([])
        for Row2 in range(Num_Of_Outputs):
            Out_Der = Outputs_Derivative[Row2]
            weight = Exterior_Weights[1][Row1,Row2]
            neuron_der[Row1].append(Out_Der * weight)
        neuron_der[Row1] = Derivative_Network[Layers-1,Row1]*sum(neuron_der[Row1]) / Rows
        #2nd To Last Layer Biases
        Gradient[2][1][Layers-1,Row1] = neuron_der[Row1] * Rows
    for Layer in range(Layers-1)[::-1]:
        temp_neuron_der = []
        for Row1 in range(Rows):
            temp_neuron_der.append([])
            for Row2 in range(Rows):
                temp_neuron_der[Row1].append(neuron_der[Row2] * Interior_Weights[Layer,Row1,Row2])
                #{Layer} Layer Weights
                Gradient[0][Layer,Row1,Row2] = neuron_der[Row2] * Network[Layer,Row1]
            temp_neuron_der[Row1] = Derivative_Network[Layer,Row1] * sum(temp_neuron_der[Row1]) / Rows
            #{Layer} Layer Biases
            Gradient[2][1][Layer,Row1] = temp_neuron_der[Row1] * Rows
        neuron_der = temp_neuron_der
    for Row1 in range(Num_Of_Inputs):
        for Row2 in range(Rows):
            #Input Weights
            Gradient[1][0][Row1,Row2] = Inputs[Row1] * neuron_der[Row2]
    return Gradient
def Gradient_Decent(Gradient,cost):
    global Interior_Weights, Exterior_Weights, Biases
    Interior_Weights -= Gradient[0]*Gradient_Magnitude*cost
    for i in [0,1]:
        Exterior_Weights[i] -= Gradient[1][i]*Gradient_Magnitude*cost
    for i in [0, 1, 2]:
        Biases[i] -= Gradient[2][i]*Gradient_Magnitude*cost
def Average_Gradients(Gradients):
    y = [[],[[],[]],[[],[],[]]]
    y[0] = sum(Gradients[0])
    y[1][0] = sum(Gradients[1][0])
    y[1][1] = sum(Gradients[1][1])
    for i in range(3):
        y[2][i] = sum(Gradients[2][i])
    return y


global Num_Of_Inputs, Num_Of_Outputs, Interior_Weights, Exterior_Weights, Biases, Network, Anti_Network, Correct_Outputs, Outputs, Inputs, Rows, Layers, Gradient_Magnitude, Gradient_Rounds

Layers = 2
Rows = 1000
Num_Of_Inputs = (len(train_X[0].ravel()))
Num_Of_Outputs = 10

Interior_Weights = rng.standard_normal((Layers - 1,Rows,Rows))
Exterior_Weights = [rng.standard_normal((Num_Of_Inputs,Rows)),rng.standard_normal((Rows,Num_Of_Outputs))]
Biases = [rng.standard_normal(Num_Of_Inputs),rng.standard_normal((Layers,Rows)),rng.standard_normal(Num_Of_Outputs)]
Biases = [np.zeros(Num_Of_Inputs),np.zeros((Layers,Rows)),np.zeros(Num_Of_Outputs)]
Network = np.zeros((Layers,Rows))
Anti_Network = np.zeros((Layers,Rows))

Gradient_Rounds = 1
Gradient_Magnitude = 1
RoundsBeforeNewInput = 1
Repeat_Until_Wrong = False

Run_Type = 'train'
while True:
    try:
        a = 'new'
        i = 1
        costs = [1]
        Gradients = [[],[[],[]],[[],[],[]]]
        b = 0
        while True:

            if a == 'new' or (a == 'repeat'):
                if Run_Type == 'train':
                    r = randrange(50000)
                    Inputs = train_X[r].ravel()/255
                    Correct_Outputs = Function(train_y[r])
                else:
                    r = randrange(1000)
                    Inputs = test_X[r].ravel()/255
                    for i1 in test_X[r]:
                        l = []
                        for i2 in i1:
                            if i2 == 0:
                                l.append(' ')
                            else:
                                l.append('#')
                        print(l)
                    Correct_Outputs = Function(test_y[r])
            Outputs = Neural_Network_Run(Inputs)
            cost = Average(Cost(Outputs,Correct_Outputs))
            costs = [cost] + costs[0:min(99,len(costs))]

            if Run_Type == 'train':
                print(cost)
            elif Run_Type == 'run':
                print(f'Output: {Outputs}\nCorrect Output: {Correct_Outputs}\nScore: {cost}')
                input()
                continue
            if not (a == 'repeat' or Run_Type == 'run') :
                a = input()
            
            i += 1

            if a == 'load':
                break
            elif a == 'run':
                Run_Type = 'run'
                break
            if not Repeat_Until_Wrong:
                b = 0
            if Gradient_Rounds == 1:
                if b == 0:
                    gradient = Gradient()
                    b = 1
                elif last_cost <= cost:
                    b = 0
                    Gradient_Decent(gradient,-cost)
                    continue
                Gradient_Decent(gradient,cost)
                last_cost = cost
                continue
            elif i%Gradient_Rounds == 0:
                if b == 0:
                    gradient = Gradient()
                    b = 1
                elif last_cost <= cost:
                    b = 0
                    Gradient_Decent(Average_Gradients(Gradients),-Average(costs[0:Gradient_Rounds-1]))
                    continue
                Gradient_Decent(Average_Gradients(Gradients),Average(costs[0:Gradient_Rounds-1]))
                last_cost = cost
                Gradients = [[],[[],[]],[[],[],[]]]
            
            
            #Gradients Append
            gradient = Gradient()
            Gradients[0].append(gradient[0])
            for I in [0,1]:
                Gradients[1][I].append(gradient[1][I])
            for I in [0,1,2]:
                Gradients[2][I].append(gradient[2][I])
    except KeyboardInterrupt:
        if a == 'repeat':
            print('Training stopped. 1 run : 2 back : 3 save : 4 settings')
            a = input('> ')
            if a == '1':
                Run_Type = 'run'
                continue
            elif a == '2':
                continue
            elif a == '3':
                a = 'y'
            elif a == '4':
                try:
                    Gradient_Magnitude = float(input(f'Learn Speed (Current {Gradient_Magnitude})> '))
                    Gradient_Rounds = int(input(f'Gradient Rounds> (Current {Gradient_Rounds})> '))
                    Repeat_Until_Wrong = bool(input(f'Repeat Until Wrong?> (Current {Repeat_Until_Wrong})> '))
                    RoundsBeforeNewInput = int(input(f'Rounds Before New Input (Current {RoundsBeforeNewInput})> '))
                except:
                    print("Invalid Input")
                a = 'repeat'
                continue
            else:
                break
        else:
            print("Program was stopped. Do you want to save the network?")
            a = input("> ")
        if a.lower() == 'y':
            a = input("name> ")+'.pkl'
            Neural_Network = [Interior_Weights, Exterior_Weights, Biases]
            with open(a,'wb') as f:
                pickle.dump(Neural_Network, f)
        break
    if a == 'load':
        a = input('name> ')
        with open(f'{a}.pkl', 'rb') as f:
            Neural_Network = pickle.load(f)
        Interior_Weights, Exterior_Weights, Biases = Neural_Network
