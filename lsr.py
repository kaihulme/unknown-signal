from __future__ import print_function # allows print from python 3 in python 2.6+
from pprint import pprint # pretty print for arrays
import sys # for command line argument fetching
import os # for system file validation
import numpy as np # for data manipulation
import matplotlib.pyplot as plt # for graph plotting

# maps functions to data points for graph plotting
def map_functions(data, coefficients):
    x_mappings = np.zeros((len(data), 100)) # creates 100 x coordinates for graph plotting for each set
    y_mappings = np.zeros((len(data), 100)) # creates 100 y coordinates for graph plotting for each set
    for i in range(len(data)): # for each set in data
        x_mappings[i] = np.linspace(min(data[i][:,0]), max(data[i][:,0]), 100) # maps 100 equally spaced points in x within range of set
        if (coefficients[i][0] == 0): # if a polynomial function
            for j, element in enumerate(y_mappings[i]): # for each y point for function mapping
                for k in range(1,(len(coefficients[i]))): # for each coefficient for set
                    y_mappings[i][j] += coefficients[i][k] * (x_mappings[i][j]**(k-1)) # apply a + bx + cx^2 + dx^3... where a,b,c... are coefficients for polynomial function
        else: # if a sinusoidal function
            for j, element in enumerate(y_mappings[i]): # for each y point for function mapping
                y_mappings[i][j] = coefficients[i][1] + (coefficients[i][2] * np.sin(x_mappings[i][j])) # apply a + bsin(X) for current x value where a and b are coefficients for sinusoidal function
    return x_mappings, y_mappings # return mapped coordinates for each function in each set in data

# plots least squares regression line
def plot_graph(data, coefficients):
    x_mappings, y_mappings = map_functions(data, coefficients) # maps points to coordinates with correct regression function
    fig, ax = plt.subplots() # creates figure to plot graph to
    for i in range(len(data)): # for each set in data
        ax.scatter(data[i][:,0], data[i][:,1]) # plot scatter of points on figure
    for i in range(len(data)): # for each set in data
        if (coefficients[i][0] == 0): # if polynomial regression
            if (coefficients[i][3] == 0): # if linear regression
                ax.plot(x_mappings[i], y_mappings[i], c='g') # plot green linear regression line
            else: # if not linear (if cubic)
                ax.plot(x_mappings[i], y_mappings[i], c='r') # plot red polynomial regression line
        else: # if not polynomial (if sinusoidal)
            ax.plot(x_mappings[i], y_mappings[i], c='b') # plot blue sinusoidal regression line
    plt.show() # show figure

#calculates sum squared error of a set of data
def sse(set, set_coefficients):
    sse = 0 # initialises current SSE to 0
    for i in range(len(set)): # for each point in set
        yi = 0 # initialises yi to 0
        if (set_coefficients[0] == 0): # if polynomial function
            for j in range(1,(len(set_coefficients))): # for all coefficients in function
                yi += set_coefficients[j] * ((set[i][0])**(j-1)) # apply a + bx + cx^2 + dx^3... where a,b,c... are coefficients for polynomial function
        else: # if a sinusoidal function
            yi = set_coefficients[1] + (set_coefficients[2] * np.sin(set[i][0])) # apply a + bsin(X) for current x value where a and b are coefficients for sinusoidal function
        sse += np.square(yi - set[i][1]) # increase current SSE by (yi-y)^2
    return sse # return the sum squared error, sum((yi-y)^2), for the given set

# calculates sum of all sum squared errors
def sum_sse(data, coefficients):
    sum = 0 # initialises sum of SSE to 0
    for i in range(len(data)): # for each set
        i_sse = sse(data[i], coefficients[i]) # get SSE for i'th set in data
        sum += i_sse # increase sum of SSE by SSE of i'th set in data
    return sum # return sum of each set's SSE in data

# calculates least squares polynomial regression line coefficients for given complexity p
def least_squares(x, y, p, max_p):
    X = np.column_stack((np.ones((x.shape[0], 1)), x)) # set first column in X to 1's and second to x column from sets data
    if (p>1): # if complexity is greater than p=1
        for i in range(1,p): # for each remaining p
            xi = np.zeros(x.shape) # initialise column of 0's the same shape as x column from sets data
            for j, element in enumerate(x): # for each x in sets data
                xi[j] = x[j]**(i+1) # set current xi to x^p for current complexity
            X = np.column_stack((X, xi)) # append xi as column on X
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # apply least squares matrix algorithm to find polynomial coefficients for given complexity p
    for i in range(p, max_p): # for each unused coefficient due to lower complexity than max_p
        A = np.append(A, 0) # append 0 so returned shape is equal for each sets function independent of complexity
    return np.append(0, A) # prepend 0 to functions coefficients to indicate polynomial function

# calculates least squares sinusoidal regression line
def least_squares_sin(x, y, max_p):
    xi = np.zeros(x.shape) # initialise column of 0's the same shape as x column from sets data
    for i, element in enumerate(x): # for each x in sets data
        xi[i] = np.sin(x[i]) # set current xi to sin(x) from sets data
    X = np.column_stack((np.ones((xi.shape[0], 1)), xi)) # set first column in X to 1's and second column to xi
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # apply lesat squares matrix algorithm to find sinusoidal coefficients
    for i in range(1, max_p): # for each unused coefficient due to lower complexity than max_p
        A = np.append(A, 0) # append 0 so returned shape is equal for each sets function independent of complexity
    return np.append(1, A) # prepend 1 to functions coefficients to indicate sinusoidal function

# % improvement of SSE from previous function to next
def diff(sse_prev, sse_next):
    return 100-((sse_next/sse_prev)*100) # calculates the accuracy improvement % in terms of SSE of one function to another

# THIS FUNCTION SHALL NOT BE CALLED - calculates coefficients for regression lines of optimal complexity without overfitting or exceeding degree of max_p
# NOTE: I think this function now doesnt work due to 0/1 now being prepended to coefficients to indicate polynomial/sinusoidal
def get_coefficients_allP(set, p, max_p, p0_coefficients, p0_sse, sse_range):
    p1_coefficients = least_squares(set[:,0], set[:,1], p+1, max_p) # get polynomial coefficients for complexity p+1
    p1_sse = sse(set, p1_coefficients) # get sum squared error for function of complexity p+1
    if (p<max_p and (diff(p0_sse, p1_sse) > sse_range)): # if error between p and p+1 is not small enough for overfitting and p+1 can be checked
        p0_coefficients = get_coefficients_allP(set, p+1, max_p, p1_coefficients, p1_sse, sse_range) # recursively call function with complexity p+1
    elif (p<(max_p-1)): # if p+2 can be checked check 2 ahead incase p and p+1 are similar but p and p+2 are not (in terms of SSE difference)
        p2_coefficients = least_squares(set[:,0], set[:,1], p+2, max_p) # get polynomial coefficients for complexity p+2
        p2_sse = sse(set, p2_coefficients) # get sum squared error for function of complexity p+2
        if (diff(p0_sse, p2_sse) > sse_range): # if error between p and p+2 is not small enough for overfitting
            if ((p+2) < (max_p-1)): # if p+3 can be checked
                p0_coefficients = get_coefficients_allP(set, p+2, max_p, p2_coefficients, p2_sse, sse_range) # recursively call function with complexity p+2
            return p2_coefficients # return coefficients for function of complexity p+2
    return p0_coefficients # return coefficients for function of complexity of p

# calculates coefficients for regression lines of optimal complexity of p=1 or p=2 without overfitting
def get_coefficients(set, p, max_p, p1_coefficients, p1_sse, sse_range):
    pS_coefficients = least_squares_sin(set[:,0], set[:,1], max_p) # gets coefficients for sinusoidal regression
    pS_sse = sse(set, pS_coefficients) # gets sum squared error for sinusoidal regression
    p3_coefficients = least_squares(set[:,0], set[:,1], p+2, max_p) # gets coefficients for polynomial regression of complexity p=3
    p3_sse = sse(set, p3_coefficients) # gets sum squared error for polynomial regression of complexity p=3
    if (diff(p1_sse, pS_sse) < sse_range): # if the difference in complexity between linear and sinusoidal regression lines are less than sse_range %
        if (diff(p1_sse, p3_sse) > sse_range): # if the difference in complexity between linear and cubic regression lines are greater than sse_range %
            return p3_coefficients # return polynomial regression coefficients of complexity p=3
        return p1_coefficients # return polynomial regression coefficients of complexity p=1
    if (diff(pS_sse, p3_sse) < sse_range): # if the difference in complexity between sinusoidal and cubic regression lines are less than sse_range %
        return pS_coefficients # return sinusoidal regression coefficients
    return p3_coefficients # return polynomial regression coefficients of complexity p=3

# gets sse and coefficients for data
def sse_handler(data):
    calcForAllP = False # if true allows regression lines of degree p to max_p, else just p=1, p=3 or sinusoidal regression
    p = 1 # start with linear regression
    max_p = 3  # maximum p - set to 2 to not allow polynomials of more than degree x^2
    sse_range = 25 # to increase models complexity accuracy improvement must be >sse_range%
    coefficients = np.zeros((len(data), max_p+2)) # coefficients for each sets regression line
    for i in range(len(data)): # for each set in data
        p0_coefficients = least_squares(data[i][:,0], data[i][:,1], p, max_p) # calculate coefficients for linear regression
        p0_sse = sse(data[i], p0_coefficients) # calculate sse for linear regression
        if (calcForAllP): get_coefficients_allP(data[i], p, max_p, p0_coefficients, p0_sse, sse_range) # for each set will calculate coefficients for regression lines of optimal complexity without overfitting or exceeding degree of max_p
        else: coefficients[i] = get_coefficients(data[i], p, max_p, p0_coefficients, p0_sse, sse_range) # for each set will calculate coefficients for regression lines of polynomial degree 1 or 3 or sinusoidal complexity without overfitting
    return sum_sse(data, coefficients), coefficients # returns the sum of all sum squared errors for each set

# extracts data from files into array of data sections
def file_handler(file):
    D = np.loadtxt(file, delimiter=',') # stores data from file into 2D array D
    return np.array([D[i * 20:(i + 1) * 20] for i in range((len(D) + 20 - 1) // 20 )]) # segments D into sets of length 20 (last set will be remaining of length <20)

# handles sse calculations and plotting calls
def data_handler(plot, file):
    data = file_handler(file) # gets data formatted by data handler
    error, coefficients = sse_handler(data) # gets total sum squared error for all sets and coefficients of regression lines for plotting
    print(error) # FINAL SSE OUTPUT
    if (plot): # if --plot was given as argument
        plot_graph(data, coefficients) # plots data with optimal regression lines

# validates command line arguments
def args_handler(args):
    if (len(args) == 1 or len(args) == 2): # ensures valid number of arguments
        if (args[0][-4:] == ".csv"): # ensures first argument is file of type .csv
            file = args[0] # sets file to first argument
            exists = os.path.isfile(file) # checks given CSV file exists
            if exists: # if the given file exists
                if (os.stat(file).st_size == 0): # checks file is not empty
                    print("ERROR: not data found in", file) # error for no data in file
                    return False, False, [] # returns arguments not valid, do not plot graph and no file data
                elif (len(args) == 1): # if file is valid and was the only argument
                    return True, False, file # returns arguments valid, do not plot graph with file data
                elif (args[1] == "--plot"): # if second argument is --plot
                    return True, True, file # returns arguments valid, plot graph with file data
            else: # if file does not exist
                print("ERROR: file", file, "not found") # error for file not found
                return False, False, [] # returns arguments not valid, do not plot graph and no file data
    print("ERROR: expected arguments of file_name.csv (+ --plot)") # error for unexpected arguments
    return False, False, [] # returns arguments not valid, do not plot graph and no file data

# calls handlers for arg validation and sse
def main(args):
    argsValid, plot, file = args_handler(args) # gets arguments validaty, whether to plot graph and file data
    if (argsValid): # if arguments are valid
        data_handler(plot, file) # handle file data
    else: # if arguments not valid
        print("ERROR: please enter valid arguments") # error for invalid arguments

# PROGRAM STARTS HERE
args = sys.argv[1:] # gets programs command line arguments
main(args) # starts main method with command line arguments
