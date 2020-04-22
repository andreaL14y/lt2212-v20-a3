# LT2212 V20 Assignment 3

PART 1 - a3_features

The script takes a directory and writes a .csv file containing the articles with dimension-reduced features (words). 
They are labeled by the foldername (author) and randomly as train or test data. Instead of copying the code from assignment 2 I put the a2.py script in the same folder and imported it in the script to use some of the functions. Furthermore, I assumed that the input directory is also in the same folder as the script. 

To run a3_features it needs the arguments input directory, the name of the outputfile which should be written and the dimension of the reduced featurelist. Optional the percentage of the test data can be set. E.g.:

	a3_features.py enron_sample output_500 500 -T 30

PART2/PART3 - a3_model.py
Here the featurefile of part one is taken, train- and test-samples are made and a Feedforward Neural Network is trained and tested. In the end the program prints the accuracy and the classification report containing precicion, recall and f1-score.

1. First the .csv featurefile is read. Training samples are constructed (get_train_samples) by taking two random documents. Their feature-values are put together in a list and it is labeled by 0 or 1 depending whether the articles are written by the same author (1) or not (0). The number of samples (trainsize) is set to 100 by default but can be changed with the option "-size" when running the program. It's secured that half of the samples are from the same author the other half is not. The test samples (get_test_samples) are made in the same way but without caring how many samples are labeled as 0 or 1. The number of samples that should be tested can be defined by the option "-testsize" it is set to 100 by default.
2. The class MyDataset is defined to prepare the data (samples) for training and testing.
3. The actual Network is set up: the data is loaded and prepared, the class Feedworward is defined. The basic model takes the inputsize (features of two documents) and gives an output of size 1 which is produced by a sigmoid. It can be run with or without a hidden layer and with or without a nonlinearity. For a nonlinearity one has the choice between 'reLU' and 'Tanh'.
4. The Network is trained for 10 epochs. The loss is computed using the BCELoss() function and I have chosen the Adam optimizer with learning rate 0.01. Optional it can be chosen a batch size for the training data, by default it's set to 1.
5. The Network is evaluated. If the output value of the model applied to the test data is greater or equal to 1 the evaluated prediciton (pred_eval) is set to 1 otherwise it's set to 0. For curiosity I let it write another csv file (named 'FeedForward_prediction.csv' containing the document-IDs, their original Label, the prediction value, the evaluated prediction and the difference between evaluated prediction and original label (either -1, 0 or 1) in a table. I Also wanted the program to print out some absolute values: how many samples have been tested and how many predictions were wrong. The percentage of wrong predictions is the same as 1-accuracy but computed manually. 
6. The Argumentparser gives several options what can be chosen by the user to run the script: 
	a) the featurefile is required in any case
	b)"-size", default=100, type=int --> The number of train-samples
	c)"-testsize", default=100, type=int --> The number of test-samples
	d)"-batchsize", default=1, type=int --> A batch size can be chosen for the training data (to train several samples within one epoch)
	e)"-hs" (hidden_size), default=0, type=int --> If a hidden layer is wanted it's size must be given with this option
	f)"-nl" (nonlinearity), default=None, type=str --> If a nonlinearity is wanted it has to be chosen here. One has the options "relu" or "tanh".

An example how to run the program with all possible options would be: 

	a3_model.py output_500 -trainsize 200 -testsize 50 -batchsize 20 -hs 100 -nl tanh


For the following evaluation I used a featurefile with 150 dimensions and 20% test instances.
Furthermore I set the batchsize to be 10, the trainsize to be 500 and the testsize to be 100 by default.

HIDDEN LAYER SIZE: none, 	NONLINEARITY: none, --> ACCURACY: 0.52, 	W-AVG: 0.75, 0.52, 0.58

HIDDEN LAYER SIZE: 5, 		NONLINEARITY: none, --> ACCURACY: 0.64, 	W-AVG: 0.80, 0.64, 0.69
HIDDEN LAYER SIZE: 5, 		NONLINEARITY: ReLU, --> ACCURACY: 0.73, 	W-AVG: 0.86, 0.73, 0.77
HIDDEN LAYER SIZE: 5, 		NONLINEARITY: Tanh, --> ACCURACY: 0.71, 	W-AVG: 0.78, 0.71, 0.73

HIDDEN LAYER SIZE: 25, 		NONLINEARITY: none, --> ACCURACY: 0.63, 	W-AVG: 0.76, 0.63, 0.67
HIDDEN LAYER SIZE: 25, 		NONLINEARITY: ReLU, --> ACCURACY: 0.89, 	W-AVG: 0.90, 0.89, 0.89
HIDDEN LAYER SIZE: 25, 		NONLINEARITY: Tanh, --> ACCURACY: 0.74, 	W-AVG: 0.80, 0.74, 0.76

HIDDEN LAYER SIZE: 75, 		NONLINEARITY: none, --> ACCURACY: 0.59, 	W-AVG: 0.72, 0.59, 0.62
HIDDEN LAYER SIZE: 75, 		NONLINEARITY: ReLU, --> ACCURACY: 0.90, 	W-AVG: 0.91, 0.90, 0.90
HIDDEN LAYER SIZE: 75, 		NONLINEARITY: Tanh, --> ACCURACY: 0.76, 	W-AVG: 0.88, 0.76, 0.79

HIDDEN LAYER SIZE: 100, 	NONLINEARITY: none, --> ACCURACY: 0.61, 	W-AVG: 0.75, 0.61, 0.66
HIDDEN LAYER SIZE: 100, 	NONLINEARITY: ReLU, --> ACCURACY: 0.84, 	W-AVG: 0.88, 0.84, 0.85
HIDDEN LAYER SIZE: 100, 	NONLINEARITY: Tanh, --> ACCURACY: 0.80, 	W-AVG: 0.87, 0.80, 0.82


I also wanted to try out a hidden layer size which is actually bigger than the feature size: 

HIDDEN LAYER SIZE: 200, 	NONLINEARITY: none, --> ACCURACY: 0.64, 	W-AVG: 0.68, 0.64, 0.66
HIDDEN LAYER SIZE: 200, 	NONLINEARITY: ReLU, --> ACCURACY: 0.85, 	W-AVG: 0.92, 0.85, 0.87
HIDDEN LAYER SIZE: 200, 	NONLINEARITY: Tanh, --> ACCURACY: 0.86, 	W-AVG: 0.89, 0.86, 0.87

It can be seen that the results are better with a hidden layer than without and better with a nonlinearity than without. Using ReLU I got slightly better values than using Tanh. The best result was an accuracy of 0.90 for a hidden layer size of 75 (50% of the featuresize) using ReLU.

PART BONUS: 
I imported a3_model and copy pasted the FNN, where I set some variables as fixed. As before I have chosen  the batchsize to be 10, the trainsize to be 500 and the testsize to be 100.
If one runs the get_plot function with the parameters "featurefile", "hidden_size_range" (which range of hidden layer sizes should be plotted), "steplength", and "outputpng"(name of the outputfile) one gets a PNG file plotting precicion and recall for the model using ReLu, Tanh and without nonlinearity. The values are computed for every "steplength". 
E.g. I ran the program as follows: 

	a3_bonus.py 'output_150', 150, 15, 'Precicion-Recall_150-15'.

Again it can be seen that the values using ReLU are best, precicision and recall using no nonlinearity are worst.




