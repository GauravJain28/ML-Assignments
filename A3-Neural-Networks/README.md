# Assignment 3: Neural Networks
The problem statement of the assignment is present [here](A3-PS.pdf).

In this problem, we will train neural networks to classify two datasets- a binary class (Toy) dataset and a multi-class (Devanagri handwritten characters) dataset.\
Toy dataset is present in this [link](https://drive.google.com/drive/folders/17eLXUEzaE1qr7X-TYT9qhAi2fMujYf61?usp=sharing).\
Devanagri handwritten characters dataset is present in this [link](https://drive.google.com/drive/folders/1OLyI4Dz_jsGz4VCX9RfuVsibON9Yl3wr?usp=sharing).

The assignment contains four parts:
- Implementation of a general neural network for binary class datasets with support for different loss functions (Cross-Entropy Loss/Mean-Square Loss) and different activation functions (Sigmoid/tanh/ReLU).\
The backpropagation algorithm  is implemented from scratch. The network is trained using Mini-Batch Gradient Descent. Support for adaptive learning rate is also present.\
Weights are initialized using Xavier Initialization.
- Implementation of a general neural network for multiclass datasets with similar features compared to the previous implementation.
- Search for the best learning rate algorithm among the following algorithms:
    - Momentum
    - Nesterov
    - RMSprop
    - Adam
    - Nadam
- Search for the best architecture for the best accuracy score on the Devanagari dataset. The search includes experimenting on the number of layers and number of neurons in each layers.

## How to run the code?
Part (a)
```
python3 neural_a.py input_path output_path param.txt
```
Here the code read the input training and test files for the toy dataset from the input_path (a directory), initialise the parameters of the network to what is provided in the param.txt file and write weights and predictions to output_path (also a directory).

Part (b)
```
python3 neural_b.py input_path output_path param.txt
```
Same as for part (a) except the input path contain files for devnagari dataset.

Part (c)
```
python3 neural_c.py input_path output_path param.txt
```
Here, the code run the specified model with the best parameters found and write the
weight files after training to the output_path in the same format as specified for part (a).\
A text file to the output_path with the name my_params.txt specifying each of the parameter in a new line for the best parameters is also generated.

```
python3 neural_c1.py input_path output_path
```
Here the code runs the two public architectures across all configurations of the parameter search space, and produce all plots/tables. Plots are produced in .png format and
tables are produced in .csv format.

Part (d)
```
python3 neural_d.py input_path output_path
```
The code run the best architecture along with
best parameters and produce the weight files to output path in the same way as
specified for part (a).\
A text file to the output_path with the name my_params.txt specifying each of the parameters in a new line for the best parameters.
```
python3 neural_d1.py input_path output_path
```
Here the code run all configurations
of the architecture and parameter search space, and produce all plots/tables to the output_path. Plots are produced in .png format and tables are produced in .csv format.

A  brief report on the different parts of the assignment is present [here](A3-Report.pdf).

## Author
* Gaurav Jain [[GitHub](https://github.com/GauravJain28/)]