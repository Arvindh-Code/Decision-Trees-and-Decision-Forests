import random
import sys
import math



'''
Yeast dataset command:
python dtree.py yeast_training.txt yeast_test.txt optimized
python dtree.py yeast_training.txt yeast_test.txt randomized
python dtree.py yeast_training.txt yeast_test.txt forest3
python dtree.py yeast_training.txt yeast_test.txt forest15

pendigits dataset command:
python dtree.py pendigits_training.txt pendigits_test.txt optimized
python dtree.py pendigits_training.txt pendigits_test.txt randomized
python dtree.py pendigits_training.txt pendigits_test.txt forest3
python dtree.py pendigits_training.txt pendigits_test.txt forest15

satellite dataset command:
python dtree.py satellite_training.txt satellite_test.txt optimized
python dtree.py satellite_training.txt satellite_test.txt randomized
python dtree.py satellite_training.txt satellite_test.txt forest3
python dtree.py satellite_training.txt satellite_test.txt forest15

'''
#reading data from the given dataset.
def read_data(filename) :
    result = []
    with open(filename, 'r') as file :
        for l in file :
            result.append([float(i) for i in l.split()])
    return result

# Calculate the entropy for the respective using log base 2
# It helps information_gain function for the gain calculation.
def calulating_entropy(distribution):
    result = 0
    sum_ = sum(distribution)
    for i in distribution:
        if i > 0:
            probability = i / sum_
            result -= probability * math.log2(probability)
    return result

# Based on the attribute and threshold the information gain has been calculated
#Its used by choose_attribute_optimized, choose_attribute_randomised for calculating the gain
def information_gain(examples, attribute, threshold):
    child_right = [example for example in examples if example[attribute] >= threshold]
    sum_ = len(examples)
    child_left = [example for example in examples if example[attribute] < threshold]
    result = calulating_entropy([example[-1] for example in examples]) - \
             (
                len(child_left) / sum_ * calulating_entropy([example[-1] for example in child_left]) +
                len(child_right) / sum_ * calulating_entropy([example[-1] for example in child_right])
             )
    return result


#Based on the option it will direct it to either choose attribute to oprimise or randomised
def choose_attribute(examples, attributes, option) :
    if option == 'optimized' :
        return choose_attribute_optimized(examples, attributes)
    elif option == 'randomized' :
        return choose_attribute_randomized(examples, attributes)

#selecting the best attribute and threshold using optimized way
# helps the DTL function to direct with the respective option
def choose_attribute_optimized(examples, attributes) :
    max_gain = -1
    best_attribute = -1
    best_threshold = -1
    for a in attributes :
        attribute_values = [example[a] for example in examples]
        l = min(attribute_values)
        m = max(attribute_values)
        for k in range(1, 51) :
            threshold = l + k * (m - l) / 51
            gain = information_gain(examples, a, threshold)
            if gain > max_gain :
                max_gain = gain
                best_attribute = a
                best_threshold = threshold
    return best_attribute, best_threshold

#selecting the best attribute and threshold using randomised way
def choose_attribute_randomized(examples, attributes) :
    best_threshold = -1
    max_gain = -1
    A = random.choice(attributes)
    attribute_values = [example[A] for example in examples]
    l = min(attribute_values)
    m = max(attribute_values)
    for k in range(1, 51) :
        threshold = l + k * (m - l) / 51
        gain = information_gain(examples, A, threshold)
        if gain > max_gain :
            max_gain = gain
            best_threshold = threshold
    return A, best_threshold

#decision tree learning - base DT model which is referred from professor's ppt
def dtl(examples, attributes, default, option) :
    if not examples :
        return default
    elif all(example[-1] == examples[0][-1] for example in examples) :
        return examples[0][-1]
    elif not attributes :
        return default
    else :
        best_attribute, best_threshold = choose_attribute(examples, attributes, option)
        tree={'attribute':best_attribute,'threshold':best_threshold}
        examples_left=[example for example in examples if example[best_attribute]<best_threshold]
        examples_right=[example for example in examples if example[best_attribute]>=best_threshold]
        # Add base case to terminate recursion
        if not examples_left or not examples_right :
            return default
        tree['left_child'] = dtl(examples_left, attributes, default, option)
        tree['right_child'] = dtl(examples_right, attributes, default, option)
        return tree

#classification of test object with the help of DT
def test_classification(tree, example):
    while isinstance(tree, dict):
        tree = tree['left_child'] if example[tree['attribute']] < tree['threshold'] else tree['right_child']
    return [tree]

#creating tree and appling the DT rule  to classify test data and returns its results.
def apply_decision_tree(tree, test_data) :
    results = []
    for i, example in enumerate(test_data) :
        predictedclass = test_classification(tree, example)
        trueclass = example[-1]
        accuracy = 1 if predictedclass[0] == trueclass else 0
        results.append({'index' : i,'predictedclass' : predictedclass[0],'trueclass' : trueclass,'accuracy' : accuracy})
    return results

#creating forest and appling the Decision forest rule  to classify test data and returns its results.
def apply_decision_forest(forest, testdata) :
    results = []
    for i, example in enumerate(testdata) :
        temp_dis = [test_classification(tree, example) for tree in forest]
        # Handle cases returned by the decision tree - None
        temp_dis = [i if i is not None else [0.0] * len(set(example[-1] for example in testdata)) for i in temp_dis]
        distribution = [dist for dist in temp_dis if dist is not None]
        if not distribution  :
            # All decision trees returned None, set a default distribution
            default = [0.0] * len(set(example[-1] for example in testdata))
            predictedclass = default.index(max(default))
            accuracy = 0
        else :
            # Add the distribution of each class, excluding None values
            sum_ = [sum(x) if all(isinstance(val, (int, float)) for val in x) else 0.0 for x in zip(*distribution)]
            # calculating the mix sum by its index
            predictedclass = sum_.index(max(sum_))
            # finding accuracy
            if predictedclass == example[-1]  :
                accuracy = 1
            else :
                accuracy = 0
        results.append({'index' : i,'predictedclass' : predictedclass,'trueclass' : example[-1],'accuracy' : accuracy})
    return results

def main() :
    if len(sys.argv) != 4 :
        print("Usage : python your_script.py training_file test_file option")
        sys.exit(1)
    option=sys.argv[3]
    if sys.argv[3] not in ['optimized','randomized','forest3','forest15'] :
        print("Invalid option. Select anyone of these - optimized,randomized,forest3,forest15.")
        sys.exit(1)
    allresult=0
    # loading the training and test data
    training_data=read_data(sys.argv[1])
    attributes=list(range(len(training_data[0]) - 1))
    test_data=read_data(sys.argv[2])
    #redirectign to forest3 or forest15 option
    if option == 'forest3' or option == 'forest15':
        if option == 'forest3':
            forest_sizes = 3
        else:
            forest_sizes = 15
        for _ in range(forest_sizes):
            tree = dtl(training_data, attributes, default=None, option='randomized')
            result = apply_decision_tree(tree, test_data)
            allresult += sum(result['accuracy'] for result in result) / len(result)
        allresult /= forest_sizes
        results = apply_decision_forest(tree, test_data)
        for result in results:
            print(f"Object Index = {result['index']}, Result = {result['predictedclass']}, True Class = {result['trueclass']}, Accuracy = {result['accuracy']}")
        print(f"\nClassification Accuracy =  {allresult}")
    # redirecting to optimized or randomized option
    elif option == 'optimized' or option == 'randomized' :
        tree = dtl(training_data, attributes, default=None, option=option)
        results = apply_decision_tree(tree, test_data)
        for result in results:
            print(f"Object Index = {result['index']}, Result = {result['predictedclass']}, True Class = {result['trueclass']}, Accuracy = {result['accuracy']}")
        print(f"\nClassification Accuracy =  {sum(result['accuracy'] for result in results) / len(results)}")
    else :
        raise ValueError("wrong command")

if __name__=="__main__" :
    main()