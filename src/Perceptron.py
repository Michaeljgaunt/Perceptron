#The Perceptron.py file contains the main function of the perceptron.

import argparse
import sys
import time
import Process

if __name__ == "__main__":
    
    #Saving the start time to time how long the perceptron takes.
    start_time = time.time()

    #Parsing the command line arguments.
    parser = argparse.ArgumentParser(description='Train and test a perceptron.', usage='%(prog)s [Mode], [Positive training file], [Negative training file], [Positive test file], [Negative test file]')
    group = parser.add_mutually_exclusive_group()
    #Adding an argument for stable mode and an argument for defined mode.
    group.add_argument("-ts", "--train_until_stable", help="Train the perceptron until the weight vector remains stable. Enter the desired number of iterations for weight vector to remain unchanged after the command.", type=int)
    group.add_argument("-tf", "--train_for", help="Train the perceptron for a manually defined number of iterations. Enter the desired number of iterations after the command.", type=int)
    #Adding an optional argument for plotting the results as a graph.
    parser.add_argument("-p", "--plot", help="Plot the results of the perceptron test on the graph. Must be used in conjunction with a training mode.", action='store_true')


    args = parser.parse_args()

    #If the correct usage is not provided, a statement is printed to the console to help the user.
    if len(sys.argv) < 3:
            print("Incorrect command line argument usage. Please refer to readme for usage documentation.")

    #If the user enters the train until stable command:
    if(args.train_until_stable):
        #Print the mode to the console.
        print "\nStable mode has been selected (" + str(args.train_until_stable) + " iterations)."

        #Calling the generate_feature_space function with the training and testing files to produce a feature space.
        print "\nGenerating feature space..."
        feature_space = Process.Feature.generate_feature_space("train.positive", "train.negative", "test.positive", "test.negative")
        #Calculating the size of the feature spac.
        feature_space_size = len(feature_space)
        print "Done."
        #Passing the feature space and feature space size to the Perceptron class in the Process.py file to be stored as class variables.
        Process.Perceptron.feature_space = feature_space
        Process.Perceptron.feature_space_size = feature_space_size
    
        #Generating the training data by creating vectors from the positive and negative training files.
        print "\nGenerating training data..."
        #The second argument is the label that will be given to the vectors.
        training_data = Process.Perceptron.generate_vectors("train.positive", 1)
        training_data.extend(Process.Perceptron.generate_vectors("train.negative", -1))
        print "Done."
    
        #Calling a function to train the perceptron. The second argument refers to how many iterations
        #the weight vector must remain stable (unchanged) before training ends, and comes from the input on the command line.
        Process.Perceptron.train_stable(training_data, args.train_until_stable)
    
        #Generating the test data by creating vectors from the positive and negative test files.
        print "\nGenerating test data..."
        #The second argument is the label that will be given to the vectors.
        test_data = Process.Perceptron.generate_vectors("test.positive", 1)
        test_data.extend(Process.Perceptron.generate_vectors("test.negative", -1))
        print "Done."
    
        #Calling a function to test the perceptron, followed by a function to print the results.
        test_results = Process.Perceptron.test(test_data)
        Process.Results.print_results(test_results)
        #If the optional plot command is input on the command line:
        if(args.plot):
            #A function is called to graph the results.
            Process.Results.graph_results(Process.Perceptron.iterations, Process.Perceptron.train_error_rate, "Graph of Train Error Rate", "Number of misclassifications", "Number of Iterations")
        #The time that the perceptron took is printed to the console.
        print "\n(" + str(round((time.time() - start_time), 1)) + " seconds)"

    #If the user enters train for {} iterations:
    if(args.train_for):
        #Print the mode to the console.
        print "\nManually defined mode has been selected (" + str(args.train_for) + " iterations)."
    
        #Calling the generate_feature_space function with the training and testing files to produce a feature space.
        print "\nGenerating feature space..."
        feature_space = Process.Feature.generate_feature_space("train.positive", "train.negative", "test.positive", "test.negative")
        #Calculating the size of the feature spac.
        feature_space_size = len(feature_space)
        print "Done."
        #Passing the feature space and feature space size to the Perceptron class in the Process.py file to be stored as class variables.
        Process.Perceptron.feature_space = feature_space
        Process.Perceptron.feature_space_size = feature_space_size
    
        #Generating the training data by creating vectors from the positive and negative training files.
        print "\nGenerating training data..."
        #The second argument is the label that will be given to the vectors.
        training_data = Process.Perceptron.generate_vectors("train.positive", 1)
        training_data.extend(Process.Perceptron.generate_vectors("train.negative", -1))
        print "Done."
    
        #Calling a function to train the perceptron. The second argument refers to how many iterations
        #of training will occur, and comes from the command line input.
        Process.Perceptron.train_defined(training_data, args.train_for)
    
        #Generating the test data by creating vectors from the positive and negative test files.
        print "\nGenerating test data..."
        #The second argument is the label that will be given to the vectors.
        test_data = Process.Perceptron.generate_vectors("test.positive", 1)
        test_data.extend(Process.Perceptron.generate_vectors("test.negative", -1))
        print "Done."
    
        #Calling a function to test the perceptron, followed by a function to print the results.
        test_results = Process.Perceptron.test(test_data)
        Process.Results.print_results(test_results)
        #If the optional plot command is input on the command line:
        if(args.plot):
            #A function is called to plot a graph of the results.
            Process.Results.graph_results(Process.Perceptron.iterations, Process.Perceptron.train_error_rate, "Graph of Train Error Rate", "Number of Iterations", "Number of misclassifications")
        #The time that the perceptron took is printed to the console.
        print "\n(" + str(round((time.time() - start_time), 1)) + " seconds)"

    
    
    
    