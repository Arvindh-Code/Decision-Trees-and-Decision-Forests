# Decision-Trees-and-Decision-Forests

## Description
This Python script implements a decision tree classifier for classification tasks. It supports both optimized and randomized attribute selection methods, as well as the creation of decision forests with options for the number of trees (forest3 or forest15).

## Usage

```bash
python dtree.py training_file test_file option
```

- **training_file**: File containing the training dataset.
- **test_file**: File containing the test dataset.
- **option**: Options for the classifier. Choose one of the following:
  - `optimized`: Decision tree with optimized attribute selection.
  - `randomized`: Decision tree with randomized attribute selection.
  - `forest3`: Decision forest with 3 trees.
  - `forest15`: Decision forest with 15 trees.

## Examples

```bash
# Decision tree with optimized attribute selection
python dtree.py yeast_training.txt yeast_test.txt optimized

# Decision tree with randomized attribute selection
python dtree.py yeast_training.txt yeast_test.txt randomized

# Decision forest with 3 trees
python dtree.py yeast_training.txt yeast_test.txt forest3

# Decision forest with 15 trees
python dtree.py yeast_training.txt yeast_test.txt forest15
```

## Datasets

The script supports multiple datasets such as yeast, pendigits, and satellite. Please refer to the dataset commands in the script comments for specific usage examples.

## Implementation Details

The script uses a decision tree learning algorithm based on information gain. The attribute selection can be optimized or randomized based on the chosen option. Decision forests are created by training multiple decision trees.

## Author

- Aravindh Gopalsamy
- gopal98aravindh@gmail.com

## License

This project is not open for external use or distribution. All rights reserved.

## Important Note for Students

**Warning:** This code is intended for educational purposes only. Please do not use this code for any assignment, and consider it as a reference implementation. Use your own implementation for academic assignments.
