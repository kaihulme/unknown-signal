# CW1: An Unknown Signal

### February, 2019

## 1 Introduction

The year is 1983, the strategy of D ́etente (de-escalation via verbal agreement)
has failed. Both the United States of America and the Soviet Union have
been stockpiling nuclear weapons and recent political events have brought
the world on the cusp of total annihilation. As an operator for a nuclear
early warning device you receive an unknown signal...

## 2 Task

In this coursework you will be given example points which follow an unknown
signal. You will be required to reconstruct the signal and give the resulting
error. Your solutions will be auto-marked with a variety of different inputs
in order to test both the correctness and robustness of your system.

## 3 Implementation

Your final solution will use the linear least squares regression code that you
have created in the worksheets which will need to be extended in order to
cope with non-linear curves. Your solution should:

- Read in a file containing a number of different line segments (each made
    up of 20 points)
- Determine the function type of line segment,e.g.linear/polynomial/unknown
    function (there is one unknown function), using least squares regression


- Produce the total reconstruction error
- Produce a figure showing the reconstructed line from the points if an
    optional argument is given.

Note: All Code written for the least squares regression sections must be
written by yourself. Any calls to Scipy’s regression solver or any other similar
library are forbidden and will be treated asnot implemented.

## 4 Marks

Each solution will go through 50 tests in order to test its correctness and
robustness. These test files will be similar in nature to the train files that
you have been provided. Each test will be worth 2 points broken down as
the following:

- 0: Solution gave incorrect output format and/or didnt finish execution.
- 1: Solution gave correct output format but a large error difference.
- 2: Solution gave correct output format and a small error difference.

Note: The error difference will be the difference in error between your
solution and a model solution we have coded. This will be different for every
test and more information will be provided when the marks are released.

### 4.1 Up to40%

Solutions which have implemented the linear least squares regression.

### 4.2 Up to60%

Solutions with an extended least squares regression to handle polynomial
inputs with few errors.

### 4.3 Up to80%

Solutions with an extended least squares regression fine-tuned to the output
signal which fail on edge cases.


### 4.4 Up to100%

A perfect solution that reconstructs every signal with minimum error.

## 5 Train/Test Files

The input files are Comma Separated Value (CSVs) which consists of two
columns for the x and y dimensions respectively. Each input file will be split
into different line segments each made up of 20 different points, so the first
20 points correspond to the first line segment, 21-40 the second line segment
etc. Two consecutive line segments may/may not follow the same trend. A
line segment will follow a linear, polynomial or another unknown function.
An example train file is given below (basic2.csv):

```
-2.9251047704944395,4.
-0.8554275589011144,8.
0.19855925795710938,10.
0.43421216060191,10.
0.624770694735732,10.
1.3614382007711343,11.
1.5943359027154818,12.
4.194060135884697,16.
4.6779257660237565,17.
5.023596843152395,18.
7.288865840413205,21.
7.485381750603618,22.
7.932320124145074,22.
9.697898711534522,25.
9.79296969415338,25.
10.338531977763036,26.
10.47247053028027,27.
11.373020960538451,28.
12.448760928878032,30.
12.518847008701732,30.42828193263826712.653923639452994,30.
12.717363846157376,30.
12.922893495415568,29.
13.82087982807715,28.
16.940698414027672,22.
17.888369346145012,20.
19.383541905873596,17.
19.669839705150725,16.
20.098166635136398,15.
20.564236539021604,15.
20.621376796845194,14.
21.505205350325678,13.
27.70770290983017,1.
28.51062072023851,-0.
28.664485990065593,-0.
29.937739282405232,-3.
30.093442528905104,-3.
30.886445731764947,-4.
31.512235111191547,-6.
31.755689686592167,-6.
```
This corresponds to 40 points of x and y co-ordinates which make up the two
different line segments of the file. Test files could consist of any number of line
segments (i.e.longer than adv3.csv) as well as have large amounts of noise.


## 6 Input/Output

The auto-marker will expect a program calledlsr.py(not a jupyter notebook)
which will be run as follows:

```
$ python lsr.py test_2.csv
```
After execution your program should print out the total reconstruction er-
ror, created by summing the error for every line segment, followed by a newline
character:

```
$ python lsr.py basic_2.csv
6.467081701001855e-
$
```
Additionally, to further understand and validate your solution we wish for your
program to be able to visualise the result given an optional parameter,--plot.
This is only to visually check that your solution is working and will not be part of
the automarker (styling of the plot itself doesn’t matter). For example, running:

```
$ python lsr.py basic_2.csv --plot
6.467081701001855e-
```
```
Will also show the following visualisation:
```

Note the presence of both the original points (blue dots) and fitted line seg-
ments (red lines).

## 7 Hints/Notes

- We have provided you a utilities file which contains two functions:loadpointsfromfile
    andviewdatapoints. The first will load the data points from a file and re-
    turn them as two numpy arrays. The second will visualise the data points
    with a different colour per segment.
- The error you need to calculate is the sum squared error.
- Regardless of the number of segments your program needs to only output
    the total reconstruction error.
- The test files will include points approximating line segments generated from
    the three functions mentioned above,i.e.linear, polynomial or the unknown
    function.
- The order of the polynomial functionwill notchange between the train
    files and the test files.
- Having a very low error might not be the correct answer to a noisy solution,
    refer to overfitting in the lecture slides.


