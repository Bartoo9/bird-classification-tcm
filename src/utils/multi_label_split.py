from skmultilearn.model_selection import iterative_train_test_split
import numpy as np

#data split for the multilabel dataset, ensuring labels are present in each split
def multi_label_split(X, y, test_size=0.1, val_size=0.1, random_state=2):

    np.random.seed(random_state)
    
    test_ratio = test_size/(1-val_size) 
    X_temp, y_temp, X_test, y_test = iterative_train_test_split(
        X, y, test_size=test_ratio)
    
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_temp, y_temp, test_size=val_size)
    
    labels_train = np.sum(y_train, axis=0) > 0
    labels_val = np.sum(y_val, axis=0) > 0
    labels_test = np.sum(y_test, axis=0) > 0
    
    missing_in_val = np.sum(~labels_val)
    missing_in_test = np.sum(~labels_test)
    
    if missing_in_val > 0 or missing_in_test > 0:
        print(f"Warning: {missing_in_val} labels missing in validation, {missing_in_test} missing in test")
    
    return X_train, X_val, X_test, y_train, y_val, y_test