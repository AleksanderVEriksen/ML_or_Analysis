import numpy as np

def sigmoid(x):
    # Numerically stable sigmoid function
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def derivativ_of_sig(output):
    x = output*(1-output)
    return x

def errorProp(input, output_target):
    return (output_target-input)

def error_Matrix(derivativ, errorProp):
    return np.multiply(derivativ,errorProp)

def calc(weights, inputs):
    return np.dot(weights, inputs)

def forwardPropagation(net, inputs):
    outputs = []
    for i in range(0,len(net)):
        bias = net[i][-1][0]
        for weights in net[i][0]:
            ss = calc(weights, inputs)
            ss+=bias
            outputs.append(sigmoid(ss))
    return outputs

def backPropagation(net, x, y, outputs):
    
    # Get output in output layer
    out_val = outputs[-1]
    new = []
    # Get outputs in nodes
    for k in range(0,len(outputs)-1):
        new.append(outputs[k])
    output_values_in_nodes = new
    print("output: ", out_val)
    #output layer, remove bias at end
    layer = net[1]
    w = []
    for n in range(0,len(layer)-1):
        for p in layer[n]:
            for f in p:
                w = p
    
    trans_mat_output = np.matrix.transpose(np.array(w))
    gradient_elem = errorProp(y,out_val)
    d_of_sig = derivativ_of_sig(out_val)
    Error_mat_output = error_Matrix(gradient_elem,d_of_sig)
    
    print("Error_output: ", Error_mat_output)
    
    #Inner layer
    mat_mul_inner = np.dot(trans_mat_output,Error_mat_output)
    error_inner = error_Matrix(mat_mul_inner, output_values_in_nodes)
    # Remove bias at the end
    layer = net[0]
    w = []
    for n in range(0,len(layer)-1):
        for p in layer[n]:
            w.append(p)
 
    trans_mat_input = np.matrix.transpose(np.array(w))
    # Input to inner layer
   
    print("Error inner: ",error_inner)
    mat_mul_input = np.matmul(error_inner,trans_mat_input)
    
    error_input = error_Matrix(mat_mul_input, x)
    print("Error_input: ", error_input)
    print("X: ", x)
    print("Target value:", y)
       
    return error_input
        

def network(input, hidden, output):
    network = []
    
    weights_for_input = [[np.random.uniform(-1,1) for i in range(input)] for i in range(hidden)]
    bias1 = [np.random.uniform(-1,1)]
    network.append([weights_for_input, bias1])
    
    weights_for_output = [[np.random.uniform(-1,1) for i in range(hidden)]for i in range (output)]
    bias2 = [np.random.uniform(-1,1)]
    network.append([weights_for_output,bias2])
    
    return network


if __name__ == "__main__":
    x = [[0,0,0,0],[1,0,0,0],[0,1,0,0],[1,1,0,0],[0,0,1,0],[1,0,1,0],[0,1,1,0],[1,1,1,0],[0,0,0,1],[1,0,0,1],[0,1,0,1],[1,1,0,1],[0,0,1,1],[1,0,1,1],[0,1,1,1],[1,1,1,1]]
    y = [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]

    net = network(4,4,1)

    outputs_layer = forwardPropagation(net, x[0])

    backPropagation(net,x[0],y[0], outputs_layer)

    