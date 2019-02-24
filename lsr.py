from __future__ import print_function
from scipy import stats
from pprint import pprint
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# maps regression function to points
def map_functions(data, coefficients):
    x_mappings = np.zeros((len(data), 100))
    y_mappings = np.zeros((len(data), 100))
    for i in range(len(data)):
        x_mappings[i] = np.linspace(min(data[i][:,0]), max(data[i][:,0]), 100)
        for j, element in enumerate(y_mappings[i]):
            y_mappings[i][j] = coefficients[i][0] + (coefficients[i][1] * x_mappings[i][j])
    return x_mappings, y_mappings

#plots linear sse regression line
def plot_linear_graph(data, coefficients):
    x_mappings, y_mappings = map_functions(data, coefficients)
    fig, ax = plt.subplots()
    for i in range(len(data)):
        ax.scatter(data[i][:,0], data[i][:,1])
    for i in range(len(data)):
        ax.plot(x_mappings[i], y_mappings[i], c='r')
    plt.show()

# calculates sum sqared error
def sse(set, set_coefficients):
    sse = 0
    for i in range(len(set)):
        sse += np.square(set_coefficients[0] + (set_coefficients[1] * set[i][0]) - set[i][1])
    return sse

# calculates sum of all sse
def sum_sse(data, coefficients):
    sum = 0
    for i in range(len(data)):
        print("sse for data set", i, ":", sse(data[i], coefficients[i]))
        sum += sse(data[i], coefficients[i])
    return sum

# calculates least squares regression line coefficients
def least_squares(x, y):
    X = np.column_stack((np.ones((x.shape[0], 1)), x))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return A[0], A[1]

# def least_squares_poly(x, y, p):
#     x = xs from data
#     y = ys from data
#     p = degree of polynomialism (linear (p=1), quadratic (p=3))
#     X = column_stack 1s and x
#     for i in p-1
#         column_stack X and x^p
#     A = least_squares_eq
#     retun a, b, c... (returns p+1 coefficients)

def file_handler(file):
    D = np.loadtxt(file, delimiter=',')
    return np.array([D[i * 20:(i + 1) * 20] for i in range((len(D) + 20 - 1) // 20 )])

# handles sse calculations and plotting calls
def sse_handler(plot, file):
    data = file_handler(file)
    coefficients = np.zeros((len(data), 2))
    for i in range(len(data)):
        coefficients[i] = least_squares(data[i][:,0], data[i][:,1])
    print("TOTAL SSE:", sum_sse(data, coefficients))
    if (plot):
        plot_linear_graph(data, coefficients)

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
        sse_handler(plot, file)
    else:
        print("ERROR: please enter valid arguments")

# gets arguments and calls main method
args = sys.argv[1:]
main(args)
