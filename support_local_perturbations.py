"""
Support file for algorithms proposed for Local Perturbations
"""

"""
Imports
"""
import numpy as np
import pandas as pd
import collections
from scipy.stats import truncnorm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import os

path = os.path.abspath('')
dataset_dir = str(path)+'/Datasets/' # Changed to directory address where dataset is located

def load_dataset(data_str,train_fraction):
    """
    Function to load all datasets according to data_str and train_fraction
    Input data_str: Name of the dataset to load
    Input train_fraction: Percentage of dataset instances to use as training dataset
    Output train_data: Training dataset
    Output train_data_target: Training dataset labels
    Output test_data: Test dataset
    Output test_data_target: Test dataset labels
    """
    if data_str == 'synth4':
        data = np.genfromtxt(dataset_dir+'/Synthetic4/synthetic4.csv',delimiter=',')
        train_data, train_data_target, test_data, test_data_target = select_train_test_synth(data,train_fraction)
    return train_data, train_data_target, test_data, test_data_target

def select_train_test_synth(data,train_fraction):
    """
    Function that splits data into train and test with their corresponding targets for the synthetic datasets
    Input data: The dataset used for the splits
    Output train_data: Training dataset
    Output train_data_target: The target of the training dataset
    Output test_data: Test dataset
    Output test_data_target: The target of the test dataset
    """
    len_data = len(data)
    range_idx = np.arange(len_data)
    np.random.shuffle(range_idx)
    train_len = int(np.round_(len_data*train_fraction))
    train_range_idx = range_idx[:train_len]
    test_range_idx = range_idx[train_len:]
    train_data = data[train_range_idx,:-1]
    train_data_target = data[train_range_idx,-1]
    test_data = data[test_range_idx,:-1]
    test_data_target = data[test_range_idx,-1]
    return train_data, train_data_target, test_data, test_data_target

def normalization_train(data,data_str):
    """
    Normalization applied to the train dataset on each feature
    Input data: Dataset to be normalized for each feature or column
    Input data_str: String of the dataset's name
    Output normalized_data: Normalized training dataset
    Output train_limits = Normalization parameters for the dataset
    """
    if data_str in ['synth4']:
        max_axis = np.max(data)
        min_axis = np.min(data)
    train_limits = np.vstack((min_axis,max_axis))
    normalized_data = (data - train_limits[0]) / (train_limits[1] - train_limits[0])
    if normalized_data.dtype == 'object':
        normalized_data = normalized_data.astype(float)
    return normalized_data, train_limits

def normalization_test(data,train_limits):
    """
    Normalization applied to the test dataset on each feature
    Input data: Dataset to be normalized for each feature or column
    Input train_limits: the maximum and minimum values per feature from the training dataset
    Output normalized_data: Normalized test dataset
    """ 
    normal_data = data
    normalized_data = (normal_data - train_limits[0]) / (train_limits[1] - train_limits[0])
    normalized_data[normalized_data < 0] = 0
    normalized_data[normalized_data > 1] = 1
    return normalized_data

def LID(x,data,k):
    """
    Function used to calculate the LID for a given instance
    Input x: Instance (can be the instane of interest or a synthetic instance)
    Input data: Training dataset
    Input k: Number of k neighbors to evaluate the LID constraint
    Output LID_val: LID value for the given instance
    Output x_kNN: kNN to the instance x of interest with regards to the training instances
    """
    laplace_add = 0.00001
    sum_val = 0
    x_kNN = knn(x,data,k)
    distance_x_kNN = [i[-1] for i in x_kNN]
    distance_x_kNN_max = distance_x_kNN[-1]
    for i in distance_x_kNN:
        sum_val += np.log((i+laplace_add)/distance_x_kNN_max)
    LID_val = (1-len(distance_x_kNN))/sum_val
    return LID_val, x_kNN

def knn(x,data,k):
    """
    Function to extract k closest instances from the training dataset with respect to instance x
    Input x: Instance (can be the instane of interest or a synthetic instance)
    Input data: Training dataset
    Input k: Number of k neighbors to evaluate the LID constraint
    Output kNN: kNN to instance x, from the training dataset
    """
    distance = []
    for i in range(len(data)):
        dist = euclidean(data[i],x)
        distance.append((data[i],dist))   
    distance.sort(key=lambda x: x[1])
    x_kNN = distance[:k]
    return x_kNN

def euclidean(x1,x2):
    """
    Calculation of the euclidean distance between two different instances
    Input x1: Instance 1
    Input x2: Instance 2
    Output euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1-x2)**2))

def LID_verification(LID_x_perturbed,epsilon,LID_x):
    """
    Function used to verify whether the generated instance is complying with the LID restriction
    Input LID_x_perturbed: LID of the instance generated as a perturbation from the instance of interest
    Input epsilon: Restriction value on the LID
    Input LID_x: LID of the instance generated as a perturbation from the instance of interest
    Output verified: Boolean variable that indicates whether the evaluated instance complies or not with the LID restriction
    """
    verified = False
    if np.abs(LID_x_perturbed - LID_x) < epsilon: 
        verified = True
    return verified

def find_freqs_laplace(index_values_tuples,train_data):
    """
    Function that finds the frequencies of all features in the dataset
    Input indices: list of indices of categorical features
    Input train_data: Training dataset
    Output freqs_all_cat_feat: Frequencies of all features in the dataset in dictionary
    """
    freqs_all_cat_feat = {}
    for i in index_values_tuples:
        freqs_i = []
        feat_index = i[0]
        feat_values = i[1]
        feat_tuple = collections.Counter(train_data[:,feat_index])
        feat_tuple_keys = list(feat_tuple.keys())
        laplace_addition = len([x for x in feat_tuple_keys if x not in feat_values])
        for j in feat_values:
            if j not in feat_tuple_keys:
                freqs_i.append(laplace_addition)
            else:
                freqs_i.append(feat_tuple[j]+laplace_addition)  
        freqs_all_cat_feat[feat_index] = np.array(freqs_i)/np.sum(freqs_i)
    return freqs_all_cat_feat

def perturbator_function(x,factor,std_dev,data_str,freqs):
    """
    Function to generate perturbances around x instance from the dataset with proper feature verification for every dataset
    Input x: Instance (can be the instane of interest or a synthetic instance)
    Input factor: Multiplier of the standard deviation of each feature
    Input std_dev: Standard deviation of the features in the dataset (training)
    Input data_str: String of the dataset's name
    Input freqs: Relative frequency of occurrence of all values in the features
    Output x_perturbed: acceptable perturbance on x instance
    """
    not_verified = True
    if data_str in ['synth4']:
        while not_verified:
            low = [0]*len(std_dev)
            up = [1]*len(std_dev)
            x_perturbed = truncnorm.rvs((low-x)/std_dev,(up-x)/std_dev,loc=x,scale=factor*std_dev,size=len(std_dev))
            not_verified = False
    elif data_str in ['Balloon']:
        while not_verified:
            feature0 = np.random.choice(np.array([0,1]),p=freqs[0])
            feature1 = np.random.choice(np.array([0,1]),p=freqs[1])
            feature2 = np.random.choice(np.array([0,1]),p=freqs[2])
            feature3 = np.random.choice(np.array([0,1]),p=freqs[3])
            x_perturbed = np.hstack((feature0,feature1,feature2,feature3))
            not_verified = False
    return x_perturbed

def single_feat_perturbation(x,factor,std_dev,data_str,freqs,feature_index):
    """
    Function that performs random vanilla perturbation on an instance on a single feature
    Input x: Instance of interest
    Input factor: Multiplier of the standard deviation of each feature
    Input std_dev: Standard deviation of the features in the dataset (training)
    Input data_str: String of the dataset's name
    Input freqs: Relative frequency of occurrence of all values in the features
    Input feature_index: Features indices to perturb the instance x on
    """
    x_perturbed = np.copy(x)
    perturbation_vector = np.zeros(len(x))
    for sel_feature in feature_index:
        not_verified = True
        if data_str in ['synth4']:
            while not_verified:
                low = 0
                up = 1
                perturbation_vector[sel_feature] = truncnorm.rvs((low-x[sel_feature])/std_dev[sel_feature],(up-x[sel_feature])/std_dev[sel_feature],loc=x[sel_feature],scale=factor*std_dev[sel_feature],size=1)
                not_verified = False
        elif data_str in ['Balloon']:
            while not_verified:
                if sel_feature == 0:
                    perturbation = np.random.choice(np.array([0,1]),p=freqs[sel_feature])
                    perturbation_vector[sel_feature] = perturbation
                elif sel_feature == 1:
                    perturbation = np.random.choice(np.array([0,1]),p=freqs[sel_feature])
                    perturbation_vector[sel_feature] = perturbation
                elif sel_feature == 2:
                    perturbation = np.random.choice(np.array([0,1]),p=freqs[sel_feature])
                    perturbation_vector[sel_feature] = perturbation
                elif sel_feature == 3:
                    perturbation = np.random.choice(np.array([0,1]),p=freqs[sel_feature])
                    perturbation_vector[sel_feature] = perturbation
                not_verified = False
        x_perturbed[sel_feature] = perturbation_vector[sel_feature]
    return x_perturbed

def single_unconstrained_pert(x,factor,std_dev,feature_index):
    """
    Function that performs unconstrained random vanilla perturbation on an instance on a single feature
    Input x: Instance of interest
    Input factor: Multiplier of the standard deviation of each feature
    Input std_dev: Standard deviation of the features in the dataset (training)
    Input data_str: String of the dataset's name
    Input freqs: Relative frequency of occurrence of all values in the features
    Input feature_index: Features indices to perturb the instance x on
    """
    x_perturbed = np.copy(x)
    for sel_feature in feature_index:
        x_perturbed[sel_feature] = np.random.normal(loc=x[sel_feature], scale=factor*std_dev[sel_feature], size=1)
    return x_perturbed

def feature_verification(x_perturbed,data_str):
    """
    Function used to verify whether the generated instance is complying with the dataset feature conditions after normalization.
    Input x_perturbed: Instance generated as a perturbation from the instance of interest.
    Input data_str: String of the dataset's name.
    Output verified: Boolean variable that indicates whether the evaluated instance complies or not with the feature conditions.
    """
    verified = False
    if data_str in ['synth0','synth1','synth2','synth3','synth4']:
        if (x_perturbed <= 1).all() and (x_perturbed >= 0).all():
            verified = True
    elif data_str in ['Balloon']:
        if np.isin(x_perturbed, [0,1]).all():
            verified = True
    return verified

def add_neighbors(x,x_perturbations,x_perturbations_dt_label,x_perturbations_lr_label,number_n,data,data_label,data_str):
    """
    Function that finds the nearest neighbors from the other class w.r.t. x 
    Input x: Instance of interest
    Input x_kNN: Nearest k neighbors to x
    Input number_n: Number of neighbors to extract
    Input data: Training dataset
    Input data_label: Training dataset labels
    Input data_str: Dataset name
    Output x_perturbations: x_perturbations with the new added instances from the opposite class
    Output x_perturbations_lr_label: Log. Reg. Label of the new x_perturbations
    Output x_perturbations_dt_label: D.T. Label of the new x_perturbations
    Output found_counter: Number of added new instances
    """
    x_opposite_class = []
    x_opposite_label = []
    class_lr = np.unique(x_perturbations_lr_label)
    class_dt = np.unique(x_perturbations_dt_label)
    found_counter = 0
    if len(class_lr) == 1 or len(class_dt) == 1:
        sorted_data = sort_data_distance(x,data,data_label)
        instances_sorted_data = [i[0] for i in sorted_data]
        instances_sorted_data_label = [i[2] for i in sorted_data]
        try_lr_mod = True
        try_dt_mod = True
        if len(class_lr) > 1:
            try_lr_mod = False
        if len(class_dt) > 1:
            try_dt_mod = False
        for j in range(len(instances_sorted_data_label)):
            class_j = instances_sorted_data_label[j]
            if try_lr_mod:
                if class_j != class_lr:
                    x_opposite_class.append(instances_sorted_data[j])
                    x_opposite_label.append(class_j)
                    found_counter += 1
            elif try_dt_mod:
                if class_j != class_dt:
                    x_opposite_class.append(instances_sorted_data[j])
                    x_opposite_label.append(class_j)
                    found_counter += 1
            if found_counter >= number_n:
                break
        if found_counter > 0:
            x_opposite_class = np.array(x_opposite_class)
            x_opposite_label = np.array(x_opposite_label)
            x_perturbations = np.vstack((x_perturbations,x_opposite_class))
            x_perturbations_dt_label = np.hstack((x_perturbations_dt_label,x_opposite_label))
            x_perturbations_lr_label = np.hstack((x_perturbations_lr_label,x_opposite_label))                            
    return x_perturbations, x_perturbations_lr_label, x_perturbations_dt_label, found_counter

def sort_data_distance(x,data,data_label):
    """
    Function to organize dataset with respect to distance to instance x
    Input x: Instance (can be the instane of interest or a synthetic instance)
    Input data: Training dataset
    Input data_label: Training dataset label
    Output data_sorted_distance: Training dataset sorted by distance to the instance of interest x
    """
    data_sorted_distance = []
    for i in range(len(data)):
        dist = euclidean(data[i],x)
        data_sorted_distance.append((data[i],dist,data_label[i]))      
    data_sorted_distance.sort(key=lambda x: x[1])
    return data_sorted_distance

def local_model_application(x_norm,x_pert,x_pert_log_reg_target,x_pert_dt_target,inconsistency_log_reg,inconsistency_dt,x_global_log_reg_prediction,x_global_dt_prediction,data_str,eps,pen):
    """
    Function that calculates the local model explanations and inconsistencies between local and global models, if any
    Input x_norm: Normalized instance of interest
    Input x_pert: Perturbation instances generated by any of the neighbor instance generation methods
    Input x_pert_log_reg_target: Target of the perturbation instances from the logistic regression method
    Input x_pert_dt_target: Target of the perturbation instances from the decision trees method
    Input inconsistency_log_reg: Variable to store the inconsistencies between local and global methods on the logistic regression model.
    Input inconsistency_dt: Variable to store the inconsistencies between local and global methods on the decision tree model
    Input x_global_log_reg_prediction: global model instance of interest prediction through logistic regression
    Input x_global_dt_prediction: global model instance of interest prediction through decision trees
    Input data_str: Dataset's name
    Input eps: Maximum depth of tree
    Input pen: Penalty for Log. Reg
    Output local_exp_num: Local numerical explanation vector
    Output local_exp_bin: Local binary explanation vector
    Output inconsistency_log_reg: updated logistic regression inconsistency variable
    Output inconsistency_dt: updated decision tree inconsistency variable
    """
    local_dt_model = DecisionTreeClassifier()
    lr_local = LogisticRegression(penalty=pen)
    x_pert_weights = perturbation_weights(x_norm,x_pert,data_str)
    local_dt_model = local_dt_model.fit(x_pert,x_pert_dt_target,sample_weight = x_pert_weights)
    local_log_reg_model = lr_local.fit(x_pert,x_pert_log_reg_target,sample_weight = x_pert_weights)
    x_local_dt_prediction = local_dt_model.predict(x_norm.reshape(1, x_pert.shape[1]))
    x_local_log_reg_prediction = local_log_reg_model.predict(x_norm.reshape(1, x_pert.shape[1]))
    if x_local_log_reg_prediction != x_global_log_reg_prediction:
        inconsistency_log_reg += 1
    if x_local_dt_prediction != x_global_dt_prediction:
        inconsistency_dt += 1
    local_exp_num = local_log_reg_model.coef_
    local_exp_bin = relevant_features_bin(local_dt_model,x_norm)
    return local_exp_num, local_exp_bin, inconsistency_log_reg, inconsistency_dt

def perturbation_weights(x,pert,data_str):
    """
    Method to calculate the weight of the perturbations generated with respect to the instance of interest
    Input x: Instance of interest
    Input pert: Perturbations
    Input data_str: String corresponding to the name of the dataset to be used
    Output weights: Distance kernel calculated weights f = e^(-D(x,z)/sigma^2). sigma^2 = np.sqrt(x.shape[1])*0.75
    """
    weights = []
    pert_distance = perturbation_distance(x,pert,data_str)
    x = x.reshape(1,-1)
    sigma = np.sqrt(x.shape[1])*0.75
    for i in pert_distance:
        weights.append(np.exp(-i[1]**2)/sigma**2)
    return weights

def perturbation_distance(x,pert,data_str):
    """
    Function that calculates the distance from the instance of interest x to the perturbations generated
    Input x: Instance of interest
    Input pert: Perturbations
    Input data_str: String corresponding to the name of the dataset to be used
    Output pert_distance: The perturbations set and their distance to the instance of interest
    """
    pert_distance = []
    if data_str in ['synth4']:
        for i in range(len(pert)):
            dist = euclidean(pert[i],x)
            pert_distance.append((pert[i],dist))
    return pert_distance

def relevant_features_bin(model,x):
    """
    Function that returns the binary feature importance vector for a set of features
    Input model: DT Model trained
    Input x: Single instance of interest
    Output features_bin: Binary vector indicating which features have been used by the DT model to classify instance x
    """
    features = []
    x = x.reshape(1,-1)
    features_bin = np.zeros(x.shape[1])
    # Based on: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
    node_indicator = model.decision_path(x)
    leaf_id = model.apply(x)
    x_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[x_id]:node_indicator.indptr[x_id + 1]]
    for node_id in node_index:
        if leaf_id[x_id] == node_id:
            continue
        else:
            features.append(model.tree_.feature[node_id])
    if not features:
        features_bin = features_bin
    else:
        features_bin[np.array(features)] = 1
    return features_bin

def feature_ranking(x,train_data,train_data_label,n_permutations,r_state,k,data_str):
    """
    Function that calculates the importance of features based on a decision tree classifier using entropy as splitting criterion
    Input x: Array of numerical values corresponding to the instance of interest
    Input train_data: Training dataset
    Input train_data_label: Training dataset label
    Input n_permutations: Repetitions of the permutations for feature importance calculation
    Input r_state: Random state for replicable results
    Input k: Number of neighbors to analyze for the results
    Input data_str: Dataset name
    Output: Feature importance
    """
    add = 0
    sorted_data_tuple = sort_data_distance(x,train_data,train_data_label)
    sorted_data_tuple_kNN = sorted_data_tuple[:k]
    sorted_data_kNN = [x[0] for x in sorted_data_tuple_kNN]
    sorted_label_kNN = [x[2] for x in sorted_data_tuple_kNN]
    while len(np.unique(sorted_label_kNN)) == 1:
        sorted_data_kNN.append(sorted_data_tuple[k+add][0])
        sorted_label_kNN.append(sorted_data_tuple[k+add][2])
        add += 1
        if add == len(sorted_data_tuple):
            break
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(sorted_data_kNN, sorted_label_kNN)
    result = permutation_importance(model, sorted_data_kNN, sorted_label_kNN, n_repeats=n_permutations,random_state=r_state)
    feature_importance = result.importances_mean
    return feature_importance