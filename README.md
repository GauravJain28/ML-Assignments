# Machine Learning Assignments
This repo contains the solutions of the assignments of the course COL341: Machine Learning offered in First (Diwali) Semester, 2021-22 at IIT Delhi.

## Assignment 1: Linear Regression
The problem statement of the assignment is present [here](./A1-Linear-Regression/A1-PS.pdf). 

In this problem, we will use Linear Regression to predict the Total Costs of a hospital patient. We have been provided with training dataset of the SPARCS Hospital dataset. The dataset is present in this [link](https://drive.google.com/drive/folders/1fSytwJZXXVFNRCO7wcVyx9l8nAPFTfZc?usp=sharing).

The assignment consists of three parts-
- Analytic solution of Linear Regression using Moore-Penrose pseudo inverse.
- Analytic solution of Ridge Regression and 10-fold cross-validation to find the optimal regularization parameter.
- Feature Generation and Feature Selection using Lasso Regression. 

The evaluation scripts are present in the [```evaluation```](./A1-Linear-Regression/evaluation/) folder. To evaluate different parts, run the following commands-

Part (a)
```
python3 grade_a.py outputfile_a.txt weightfile_a.txt model_outputfile_a.txt model_weightfile_a.txt
```
Part (b)
```
python3 grade_b.py outputfile_b.txt weightfile_b.txt model_outputfile_b.txt model_weightfile_b.txt
```

A  brief report on the feature generation and feature selection is present [here](./A1-Linear-Regression/A1-Report.pdf).
## Assignment 2: Logistic Regression
