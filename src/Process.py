#The Process.py file contains the functions for implementing the perceptron.

import numpy 
import random
import matplotlib.pyplot as graph

#The Feature class contains functions that are to do with features and feature
#spaces.
class Feature:
    
    #Defining a function to get the feature set from a file.
    @staticmethod
    def get_featureset(filename):

        #Instantiating feature_set variable as a set.
        feature_set = set()
        #Opening the file, and for each line, delimiting features by spaces and removing the newline character.
        with open(filename) as file:
            for line in file:
                for w in line.strip().split():	
                    #Adding each feature to the feature set.
                    feature_set.add(w)

        #Closing the file and returning the feature set.
        return feature_set
    
    #Defining a function to generate a feature space from training and test files.
    @staticmethod
    def generate_feature_space(positive_training_file, negative_training_file, positive_test_file, negative_test_file):

        #Defining feature_space as the feature set from the positive training file.
        feature_space = Feature.get_featureset(positive_training_file)
        #Unioning the feature set from the positive file with the feature set from the negative file.
        feature_space = feature_space.union(Feature.get_featureset(negative_training_file))
        #Unioning the feature space with the feature sets in the two test files.
        feature_space = feature_space.union(Feature.get_featureset(positive_test_file))
        feature_space = feature_space.union(Feature.get_featureset(negative_test_file))
        #Saving the feature space size and converting the feature space to a list.
        feature_space = list(feature_space)
        #Instantiating a dictionary to hold the feature space.
        feature_space_dict = {}
        #Iterating through the feature space list and putting it into a dictionary.
        for (feature_id, feature_value) in enumerate(feature_space):
            feature_space_dict[feature_value] = feature_id
        #Returning the feature space dictionary and the size of the feature space.
        return feature_space_dict


#The perceptron class contains functions that are to do with generating vectors,
#training and testing the perceptron.
class Perceptron:
    
    #Initializing class variables.
    feature_space = 0
    feature_space_size = 0
    weight_vector = []
    bias = 0
    
    #Instantiating lists to hold values to allow a graph of train error rate and test error rate to be plotted.
    train_error_rate = []
    test_error_rate = []
    iterations = []

    
    #Defining a function to generate a set of data as a vector.
    @staticmethod
    def generate_vectors(filename, label):
        
        #Instantiating a list to store all of the review vectors.
        review_vectors = []
        #Opening the file passed in the function arguments.
        with open(filename) as file:
            #For each line in the file,
            for line in file:
                #A temporary list is made to hold the current vector.
                temp_vector = numpy.zeros(Perceptron.feature_space_size, dtype=float)
                #For each word on the current line, features are delimited by spaces and split.
                for word in line.strip().split():
                    #The feature space index of the currently iterating word is saved
                    feature_index = Perceptron.feature_space[word]
                    #The element in the temporary vector at that index is changed to a 1 to indicate the word's presence in the review.
                    temp_vector[feature_index] = 1
                #The temporary vector is appended as an entry in the review_vector list.
                review_vectors.append((temp_vector, label))
        #The review vector list is returned.
        return review_vectors
    
    #Defining a function to train the perceptron until the weight vector remains stable (unchanged) for a defined number
    #of iterations. Following this, the function returns the ideal weight vector and bias value.
    @staticmethod
    def train_stable(training_data, stability):
    
        #Storing the length of the training data.
        training_data_length = len(training_data)

        #Initializing the activation, bias and weight vector.
        activation = 0
        bias = 0
        weights = numpy.zeros(Perceptron.feature_space_size)
        #Initializing an array to hold all the weight vectors to average later.
        iterated_weights = []
        #Instantiating the counters.
        misclassification_counter = 0
        stability_counter = 0
        loop_counter = 0

        print "\nBeginning training sequence..."
        print "Training will finish when the weight vector remains stable (unchanged) for " + str(stability) + " iteration(s)."
        #While the stability counter is less than 3,
        while(stability_counter < stability):      
            print "Iteration " + str(loop_counter + 1) + "."
            if(loop_counter != 0):
                #If no misclassifications were made on the previous iteration:
                if(misclassification_counter == 0):
                    #The stability counter is incremented.
                    stability_counter += 1
                #If any misclassifications were made:
                else:
                    #The stability counter is reset.
                    stability_counter = 0
                #The misclassification counter is reset for the coming iteration:
                misclassification_counter = 0
            #The training data is shuffled.
            random.shuffle(training_data)
            #iterating over the length of the training data.
            for j in xrange(0, training_data_length):
                #Calculating the activation by computing the dot product of the weight vector with the current review vector, and adding the bias.
                activation = (numpy.dot(weights, training_data[j][0])) + bias
                #If activation * the label is <= 0, a misclassification has occurred:
                if((activation * training_data[j][1]) <= 0):
                    #The misclassification counter is incremented.
                    misclassification_counter += 1
                    #The weight values are altered by adding the product of the word value and the review label.
                    weights += training_data[j][1] * training_data[j][0]
                    #The bias is altered by adding the review label.
                    bias += training_data[j][1]
            #The weight vector is added into another list to be averaged later.
            iterated_weights.append(weights)
            #Increasing the loop counter.
            loop_counter += 1
            print str(misclassification_counter) + " misclassifications."
        print "Weight vector has stabilized, training complete."
        
        #Iterating over the number of training iterations:
        print "\nAveraging weight vectors..."
        for i in xrange(0, loop_counter):
            #If it is the first iteration:
            if i == 0:
                #The average weight vector is set as the first iterated weight vector.
                averaged_weight_vector = iterated_weights[i]
            #If it is any other iteration:
            else:
                #The average weight vector has the currently iterating weight vector added to it.
                averaged_weight_vector = numpy.add(averaged_weight_vector, iterated_weights[i])
        #The list is divided by the number of iterations to give the final average weight vector
        averaged_weight_vector /= loop_counter
        
        #Both the finished weight vector and bias value are saved.
        Perceptron.weight_vector.append(averaged_weight_vector)
        Perceptron.bias += bias
        print "Done."

    #Defining a function to train the perceptron for a defined number of iterations. Following this, the function returns 
    #the weight vector and bias value obtained after the number of iterations.
    @staticmethod
    def train_defined(training_data, iterations, plotFlag):
            
        #Storing the length of the training data.
        training_data_length = len(training_data)

        #Initializing the activation, bias and weight vector.
        activation = 0
        bias = 0
        weights = numpy.zeros(Perceptron.feature_space_size)
        #Initializing an array to hold all the weight vectors to average later.
        iterated_weights = []
        #Instantiating the counters.
        misclassification_counter = 0

        print "\nBeginning training sequence (" + str(iterations) + " iteration(s))..."
        #While the stability counter is less than 3,
        for i in xrange(0, iterations):   
            print "Iteration " + str(i + 1) + "."   
            #If it is the first iteration:
            if(i == 0):
                #If the plotFlag is true:
                if(plotFlag):
                    #Append the train error rate and number of iterations to their corresponding arrays.
                    Perceptron.train_error_rate.append(100)
                    Perceptron.iterations.append(i)
            #If it is any other iteration:
            else:
                #If the plotFlag is true
                if(plotFlag):
                    #The missclassification counter is added into the train error rate list for plotting. 
                    Perceptron.train_error_rate.append(float(100) - ((float(training_data_length) - float(misclassification_counter)) / float(training_data_length)) * float(100))
                    #The loop counter is added into the iterations list for plotting a graph later.
                    Perceptron.iterations.append(i)
                #The misclassification counter is reset for the coming iteration:
                misclassification_counter = 0
            #The training data is shuffled.
            random.shuffle(training_data)
            #iterating over the length of the training data.
            for j in xrange(0, training_data_length):
                #Calculating the activation by computing the dot product of the weight vector with the current review vector, and adding the bias.
                activation = (numpy.dot(weights, training_data[j][0])) + bias
                #If activation * the label is <= 0, a misclassification has occurred:
                if((activation * training_data[j][1]) <= 0):
                    #The misclassification counter is incremented.
                    misclassification_counter += 1
                    #The weight values are altered by adding the product of the word value and the review label.
                    weights += training_data[j][1] * training_data[j][0]
                    #The bias is altered by adding the review label.
                    bias += training_data[j][1]                    
            #The weight vector is added into another list to be averaged later.
            iterated_weights.append(weights)
            print str(misclassification_counter) + " misclassifications."
                
        print "Training complete."
        
        #Iterating over the number of training iterations:
        print "\nAveraging weight vectors..."
        for i in xrange(0, iterations):
            #If it is the first iteration:
            if i == 0:
                #The average weight vector is set as the first iterated weight vector.
                averaged_weight_vector = iterated_weights[i]
            #If it is any other iteration:
            else:
                #The average weight vector has the currently iterating weight vector added to it.
                averaged_weight_vector = numpy.add(averaged_weight_vector, iterated_weights[i])
        #The list is divided by the number of iterations to give the final average weight vector
        averaged_weight_vector /= iterations
        
        #Both the finished weight vector and bias value are saved.
        Perceptron.weight_vector.append(averaged_weight_vector)
        Perceptron.bias += bias
        print "Done."        
        
    #Defining a function to test the perceptron on a set of annotated data.
    @staticmethod    
    def test(test_data):

        #Saving the length of the test data and intantiating the test_results list.
        test_data_length = len(test_data)
        test_results = []
        for i in xrange(0, test_data_length):
            test_results.append([])

        print "\nAnalyzing test data..."
        #Iterating over the length of the test data.
        for j in xrange(0, test_data_length):
            #Calculating the activation by taking the dot product of the trained weight vector with each review vector and adding the bias.
            activation = numpy.dot(Perceptron.weight_vector, test_data[j][0]) + Perceptron.bias
            #Saving the sign of the activation as the first element in the results list.
            test_results[j].append(numpy.sign(activation))
            #Saving the actual label of the review as the second element in the results list.
            test_results[j].append(test_data[j][1])
        print "Done."
        #Returning the results list.
        return test_results
 
#The print class contains the print results function.
class Results:
    
    #Defining a function to print the results of the perceptron testing to the console.
    @staticmethod
    def print_results(test_results):
    
        #Initializing counters and the length of the test results list.
        test_results_length = len(test_results)
        pos_counter = 0
        neg_counter = 0
        pos_correct_counter = 0
        neg_correct_counter = 0

        #Iterating over the length of the test results list.
        for i in xrange (0, test_results_length):
            #If the current element is a positive test review, and the predicted label matches:
            if((test_results[i][1] == 1) and (test_results[i][1] == test_results[i][0])):
                #The total positive counter is incremented, and the correct counter is incremented.
                pos_counter += 1
                pos_correct_counter += 1
            #If the current element is a positive test review but the predicted label does not match:
            elif((test_results[i][1] == 1) and (test_results[i][1] != test_results[i][0])):
                #Only the total positive counter is incremented.
                pos_counter += 1
            #If the current element is a negative test review and the predictd label matches:
            elif((test_results[i][1] == -1) and (test_results[i][1] == test_results[i][0])):
                #The total negative counter is incremented, and the correct counter is incremented.
                neg_counter += 1
                neg_correct_counter += 1
            #If the current element is a negative test review but the predicted label does not match:
            elif((test_results[i][1] == -1) and (test_results[i][1] != test_results[i][0])):
                #Only the negative counter is incremented.
                neg_counter += 1

        #The percentages are calculated.
        pos_correct_percentage = (float(pos_correct_counter) * 100) / (float(pos_counter))
        neg_correct_percentage = (float(neg_correct_counter) * 100) / (float(neg_counter))
        tot_correct_percentage = (float(pos_correct_counter + neg_correct_counter) * 100) / (float(pos_counter + neg_counter))
        
        #The results are printed to the console.
        print "\nThe perceptron correctly predicted " + str(round(pos_correct_percentage, 1)) + "% of positive test reviews."
        print "The perceptron correctly predicted " + str(round(neg_correct_percentage, 1)) + "% of negative test reviews."
        print "The perceptron correctly predicted " + str(round(tot_correct_percentage, 1)) + "% of all test reviews."

    #Defining a function to count the results of the perceptron.
    @staticmethod
    def count_results(test_results):
    
        #Initializing counters and the length of the test results list.
        test_results_length = len(test_results)
        pos_counter = 0
        neg_counter = 0
        pos_correct_counter = 0
        neg_correct_counter = 0

        #Iterating over the length of the test results list.
        for i in xrange (0, test_results_length):
            #If the current element is a positive test review, and the predicted label matches:
            if((test_results[i][1] == 1) and (test_results[i][1] == test_results[i][0])):
                #The total positive counter is incremented, and the correct counter is incremented.
                pos_counter += 1
                pos_correct_counter += 1
            #If the current element is a positive test review but the predicted label does not match:
            elif((test_results[i][1] == 1) and (test_results[i][1] != test_results[i][0])):
                #Only the total positive counter is incremented.
                pos_counter += 1
            #If the current element is a negative test review and the predictd label matches:
            elif((test_results[i][1] == -1) and (test_results[i][1] == test_results[i][0])):
                #The total negative counter is incremented, and the correct counter is incremented.
                neg_counter += 1
                neg_correct_counter += 1
            #If the current element is a negative test review but the predicted label does not match:
            elif((test_results[i][1] == -1) and (test_results[i][1] != test_results[i][0])):
                #Only the negative counter is incremented.
                neg_counter += 1

        #The percentage correct is calculated.
        tot_incorrect_percentage = float(100) - ((float(pos_correct_counter + neg_correct_counter) * 100) / (float(pos_counter + neg_counter)))
        
        return tot_incorrect_percentage

    #Defining a method to print the results on a grap.
    @staticmethod
    def graph_results(xaxis, yaxis, yaxis2, title, xlabel, ylabel):
        #Plotting datasets onto the xaxis and y-axis.
        print "\nBuilding graph..."
        graph.plot(list(xaxis), list(yaxis), list(xaxis), list(yaxis2)) 
        graph.title(title)
        graph.xlabel(xlabel)
        graph.ylabel(ylabel)
        graph.show()
        print "Done."