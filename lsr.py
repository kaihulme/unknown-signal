from __future__ import print_function
from scipy import stats
from pprint import pprint
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# maps points on least squares regression function
# def map_functions(data, coefficients):
#     x_mappings = np.zeros((len(data), 100))
#     y_mappings = np.zeros((len(data), 100))
#     for i in range(len(data)):
#         x_mappings[i] = np.linspace(min(data[i][:,0]), max(data[i][:,0]), 100)
#         for j, element in enumerate(y_mappings[i]):
#             y_mappings[i][j] = coefficients[i][0]
#             for k in range(1,(len(coefficients[i]))):
#                 y_mappings[i][j] += coefficients[i][k] * (x_mappings[i][j]**k)
#     return x_mappings, y_mappings

def map_functions(data, coefficients):
    x_mappings = np.zeros((len(data), 100))
    y_mappings = np.zeros((len(data), 100))
    for i in range(len(data)):
        x_mappings[i] = np.linspace(min(data[i][:,0]), max(data[i][:,0]), 100)
        if (coefficients[i][0]):
            for j, element in enumerate(y_mappings[i]):
                for k in range(1,(len(coefficients[i]))):
                    y_mappings[i][j] += coefficients[i][k] * (x_mappings[i][j]**k)
        else:
            for j, element in enumerate(y_mappings):
                y_mappings[j] = coefficients[i][0] + (coefficients[i][0] * math.sin(x_mappings[j]))
    return x_mappings, y_mappings

# plots least squares regression line
def plot_graph(data, coefficients):
    x_mappings, y_mappings = map_functions(data, coefficients)
    fig, ax = plt.subplots()
    for i in range(len(data)):
        ax.scatter(data[i][:,0], data[i][:,1])
    for i in range(len(data)):
        ax.plot(x_mappings[i], y_mappings[i], c='r')
    plt.show()

# calculates sum squared error of a set of data
# def sse(set, set_coefficients):
#     sse = 0
#     for i in range(len(set)):
#         yi = set_coefficients[0]
#         for j in range(1,(len(set_coefficients))):
#             yi += set_coefficients[j] * ((set[i][0])**j)
#         sse += np.square(yi - set[i][1])
#     return sse

#calculates sum squared error of a set of data
def sse(set, set_coefficients):
    sse = 0
    yi = 0
    for i in range(len(set)):
        yi = 0
        if (set_coefficients[0] == 0):
            for j in range(0,(len(set_coefficients))):
                yi += set_coefficients[j] * ((set[i][0])**j)
        else:
            yi = set_coefficients[1] + (set_coefficients[1] * np.sin(set[i][1]))
        sse += np.square(yi - set[i][1])
    return sse

# calculates sum of all sum squared errors
def sum_sse(data, coefficients):
    sum = 0
    for i in range(len(data)):
        i_sse = sse(data[i], coefficients[i])
        sum += i_sse
    return sum

# calculates least squares regression line coefficients for given p
def least_squares(x, y, p, max_p):
    X = np.column_stack((np.ones((x.shape[0], 1)), x))
    if (p>1):
        for i in range(1,p):
            xi = np.zeros(x.shape)
            for j, element in enumerate(x):
                xi[j] = x[j]**(i+1)
            X = np.column_stack((X, xi))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    for i in range(p, max_p):
        A = np.append(A, 0)
    return np.append(0, A)

def least_squares_sin(x, y, max_p):
    xi = np.zeros(x.shape)
    for i, element in enumerate(x):
        xi[i] = np.sin(x[i])
    X = np.column_stack((np.ones((xi.shape[0], 1)), xi))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    for i in range(1, max_p):
        A = np.append(A, 0)
    return np.append(1, A)

# get coefficients for set with optimal p
def get_coefficients(set, p, max_p, p0_coefficients, p0_sse):                                   # recursively compare error differences between sse for current p and (p+1) or (p+2)
    sse_range = 10                                                                              # error difference to check for
    p1_coefficients = least_squares(set[:,0], set[:,1], p+1, max_p)                             # get coefficients for p+1
    p1_sse = sse(set, p1_coefficients)                                                          # get sse for p+1

    #pprint(p0_coefficients)

    if (p<max_p and (p0_sse/p1_sse > sse_range)):                                               # if error between p and p+1 is not small enough for overfitting and p+1 can be checked
        p0_coefficients = get_coefficients(set, p+1, max_p, p1_coefficients, p1_sse)            # compare p+1 and p+2
    elif (p<(max_p-1)):                                                                         # if p+2 can be checked check 2 ahead incase p and p+1 are similar but p and p+2 are not
        p2_coefficients = least_squares(set[:,0], set[:,1], p+2, max_p)                         # get coefficients for p+2
        p2_sse = sse(set, p2_coefficients)                                                      # get sse for p+2
        if (p0_sse/p2_sse > sse_range):                                                         # if error between p and p+2 is not small enough for overfitting
            if ((p+2) < (max_p-1)):                                                             # if p+3 can be checked
                p0_coefficients = get_coefficients(set, p+2, max_p, p2_coefficients, p2_sse)    # compare p+2 and p+3
            return p2_coefficients                                                              # return coefficients for p+2
    return p0_coefficients                                                                      # return coefficients for p

# gets sse and coefficients for data
def sse_handler(p, max_p, data):
    # coefficients = np.zeros((len(data), max_p+1))
    coefficients = np.zeros((len(data), max_p+2))
    for i in range(len(data)):

        p0_coefficients = least_squares(data[i][:,0], data[i][:,1], p, max_p)                   # calculate coefficients for linear
        p0_sse = sse(data[i], p0_coefficients)                                                  # calculate sse for linear
        coefficients[i] = get_coefficients(data[i], p, max_p, p0_coefficients, p0_sse)          # return coefficients with least error & non-overfitting

        print("polynomial sse:", sse(data[i], coefficients[i]))

        sin_coefficients = least_squares_sin(data[i][:,0], data[i][:,1], max_p)
        sin_sse = sse(data[i], sin_coefficients)

        print("sin sse", sin_sse)

        if (sin_sse < sse(data[i], coefficients[i])):
            coefficients[i] = sin_coefficients

    return sum_sse(data, coefficients), coefficients

# extracts data from files into array of data sections
def file_handler(file):
    D = np.loadtxt(file, delimiter=',')
    return np.array([D[i * 20:(i + 1) * 20] for i in range((len(D) + 20 - 1) // 20 )])

# handles sse calculations and plotting calls
def data_handler(plot, file):
    data = file_handler(file)
    start_p = 1 # start with linear regression
    max_p = 5  # maximum p to stop overflow errors
    error, coefficients = sse_handler(start_p, max_p, data)
    #print("TOTAL SSE: ",error)
    print(error)
    if (plot):
        plot_graph(data, coefficients)

# validates command line arguments
def args_handler(args):
    if (len(args) == 1 or len(args) == 2):
        if (args[0][-4:] == ".csv"):
            file = "train/" + args[0]
            exists = os.path.isfile(file)
            if exists:
                if (os.stat(file).st_size == 0):
                    print("ERROR: not data found in", file)
                    return False, False, []
                elif (len(args) == 1):
                    return True, False, file
                elif (args[1] == "--plot"):
                    return True, True, file
            else:
                print("ERROR: file", file, "not found")
                return False, False, []
    print("ERROR: expected arguments of file_name.csv (+ --plot)")
    return False, False, []

# calls handlers for arg validation and sse
def main(args):
    argsValid, plot, file = args_handler(args)
    if (argsValid):
        data_handler(plot, file)
    else:
        print("ERROR: please enter valid arguments")

# gets arguments and calls main method
args = sys.argv[1:]
main(args)
