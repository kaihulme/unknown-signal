from __future__ import print_function
from scipy import stats
from pprint import pprint
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# maps points on least squares regression function
def map_functions(data, coefficients):
    x_mappings = np.zeros((len(data), 100))
    y_mappings = np.zeros((len(data), 100))
    for i in range(len(data)):
        x_mappings[i] = np.linspace(min(data[i][:,0]), max(data[i][:,0]), 100)
        for j, element in enumerate(y_mappings[i]):
            y_mappings[i][j] = coefficients[i][0]
            for k in range(1,(len(coefficients[i]))):
                y_mappings[i][j] += coefficients[i][k] * (x_mappings[i][j]**k)
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
def sse(set, set_coefficients):
    sse = 0
    for i in range(len(set)):
        yi = set_coefficients[0]
        for j in range(1,(len(set_coefficients))):
            yi += set_coefficients[j] * ((set[i][0])**j)
        sse += np.square(yi - set[i][1])
    return sse

# calculates sum of all sum squared errors
def sum_sse(data, coefficients):
    sum = 0
    for i in range(len(data)):
        i_sse = sse(data[i], coefficients[i])
        print("sse for data set", i, ":", i_sse)
        sum += i_sse
    return sum

# calculates least squares regression line coefficients for given p
def least_squares(x, y, p, max_p):
    X = np.column_stack((np.ones((x.shape[0], 1)), x))
    if (p>1):
        for i in range(1,p):
            x_exp = np.zeros(x.shape)
            for j, element in enumerate(x):
                x_exp[j] = x[j]**(i+1)
            X = np.column_stack((X, x_exp))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    for i in range(p, max_p):
        A = np.append(A, 0)
    return A

# get coefficients for set with optimal p
def get_coefficients(set, p, max_p, p0_coefficients, p0_sse):
    sse_range = 1.1
    p1_coefficients = least_squares(set[:,0], set[:,1], p+1, max_p)
    p1_sse = sse(set, p1_coefficients)
    if (p<max_p and (p0_sse/p1_sse > sse_range)):
        p0_coefficients = get_coefficients(set, p+1, max_p, p1_coefficients, p1_sse)
    elif (p<(max_p-2)):
        p2_coefficients = least_squares(set[:,0], set[:,1], p+2, max_p)
        p2_sse = sse(set, p2_coefficients)
        if (p0_sse/p2_sse > sse_range):
            p0_coefficients = get_coefficients(set, p+2, max_p, p2_coefficients, p2_sse)
    return p0_coefficients

# gets sse and coefficients for data
def sse_handler(p, max_p, data):
    coefficients = np.zeros((len(data), max_p+1))
    for i in range(len(data)):
        p0_coefficients = least_squares(data[i][:,0], data[i][:,1], p, max_p)                  # calculate coefficients for linear
        p0_sse = sse(data[i], p0_coefficients)                                          # calculate sse for linear
        coefficients[i] = get_coefficients(data[i], p, max_p, p0_coefficients, p0_sse)            # return coefficients with least error & non-overfitting
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
    print("TOTAL SSE: ",error)
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
