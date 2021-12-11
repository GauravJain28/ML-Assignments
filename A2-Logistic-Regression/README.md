# Assignment 2: Logistic Regression
The problem statement of the assignment is present [here](A2-PS.pdf). 

In this problem, we will use Logistic Regression to build a classifier for Hospital Inpatient Discharges training data. \
We build a logistic regression model for 8 class classification to predict the ’Length
of Stay’ (target column in the given dataset) of the patients in the Hospital. The length of Stay can be any of the one among {1,2,3,4,5,6,7,8} days (here label 8 means ≥ 8 days). \
The dataset is present in this [link](https://drive.google.com/drive/folders/1eUrTWOAoVQ_LiPHBxe_zLO2whL0sQvp1?usp=sharing).

The assignment consists of four parts-
1. Implementation of Gradient Descent Algorithm with 3 learning strategies:
    - Constant learning rate 
    - Adaptive learning rate 
    - Adaptive learning rate using αβ backtracking line search algorithm 
2. Implementation of Mini-batch Gradient Descent Algorithm with 3 learning strategies:
    - Constant learning rate 
    - Adaptive learning rate 
    - Adaptive learning rate using αβ backtracking line search algorithm 
3. Selection of best algorithm and learning strategy for logistic regression and hyperparameter optimization.
4. Feature generation and feature selection using k-fold cross-validation.

## How to run the code?
```
python3 logistic.py Mode Parameters
```
The Mode corresponds to part [a,b,c,d] of the assignment.

The parameters are dependant on the mode:

Part (a)
```
python3 logistic.py a trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt
```
The predictions are stored in a line aligned ```outputfile.txt```. Also weight matrix (which includes bias terms in the first row) by flattening the weight matrix row wise is stored in a line aligned ```weightfile.txt```. \
Here, ```param.txt``` contain three lines of input, the first being a number [1-3] indicating which learning rate strategy to use and the second being parameters of the strategy. The third line will be the exact number of iterations.

Part (b)
```
python3 logistic.py b trainfile.csv testfile.csv param.txt outputfile.txt weightfile.txt
```
The arguments mean the same as mode a, with an additional line 4 in ```param.txt``` specifying the batch size and also the third line is the exact number of epochs to run on.

Part (c)
```
python3 logistic.py c trainfile.csv testfile.csv outputfile.txt weightfile.txt
```
Best learning strategy and best hyperparameters are used.

Part (d)
```
python3 logistic.py d trainfile.csv testfile.csv outputfile.txt weightfile.txt
```
Additional features are generated and selected for higher accuracy scores.

## Evaluation
The evaluation scripts are present in the [```evaluation```](evaluation/) folder. There are 8 testcases in the folder. To evaluate different testcases, run the following commands-

Part (a) [Testcases 1,2,3,7]
```
python3 grade_a.py outputfile_a.txt weightfile_a.txt model_outputfile_a.txt model_weightfile_a.txt
```

Part (b) [Testcases 4,5,6,8]
```
python3 grade_b.py outputfile_b.txt weightfile_b.txt model_outputfile_b.txt model_weightfile_b.txt
``` 

A  brief report on the different parts of the assignment is present [here](A2-Report.pdf).

## Author
* Gaurav Jain [[GitHub](https://github.com/GauravJain28/)]
