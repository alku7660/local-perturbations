"""
Algorithms proposed for Local Perturbations
"""

"""
Imports
"""
from support_local_perturbations import *

"""
Feature Vanilla Random Algorithm (Feat. Van. R.)
"""
def feat_perturbator_vanilla(x,data,std_dev,LID_x,data_str,factor,N,k,freqs,validity = None):
    """
    Function used for random generation of new instances without considering LID restriction
    Input x: Array of numerical values corresponding to the instance of interest
    Input data: Dataset used
    Input std_dev: Standard deviation of the features in the data
    Input LID_x: LID of the instance of interest
    Input data_str: String of the dataset's name
    Input factor: Multiplier of the standard deviation of each feature
    Input N: The number of instances to generate complying with the LID restriction
    Input k: Number of k neighbors to evaluate the LID constraint
    Input freqs: Frequencies of the different values in the dataset for all features
    Input validity: When None, no valid perturbation method called
    Output x_perturbations_array: Array of instances generated from the perturbation
    Output LID_perturbations_array: Array of LID values of the complying generated instances generated from the perturbation
    Output verified_instances: Number of valid instances
    """
    LID_ft = int(LID_x)
    x_perturbations = []
    LID_perturbations = []
    number_instances = 0
    verified_instances = 0
    if LID_ft > len(x):
        LID_ft = len(x)
    while number_instances < N:
        feature_index = np.random.choice(range(len(std_dev)),size=(int(LID_ft)),replace=False)
        if validity == 'yes':
            x_perturbed = single_feat_perturbation(x,factor,std_dev,data_str,freqs,feature_index)
        else:
            x_perturbed = single_unconstrained_pert(x,factor,std_dev,feature_index)
        LID_x_perturbed, x_perturbed_kNN = LID(x_perturbed,data,k)
        x_perturbations.append(x_perturbed)
        LID_perturbations.append(LID_x_perturbed)
        number_instances += 1
        verified = feature_verification(x_perturbed,data_str)
        if verified:
            verified_instances += 1
    x_perturbations_array = np.array(x_perturbations)
    LID_perturbations_array = np.array(LID_perturbations)
    return x_perturbations_array, LID_perturbations_array, verified_instances

"""
Feature Vanilla Random with Feature Importance (Feat. Van. R. FI)
"""
def feat_perturbator_vanilla_fi(x,data,data_label,std_dev,LID_x,data_str,factor,N,k,freqs,validity = None):
    """
    Function used for random generation of new instances without considering LID restriction
    Input x: Array of numerical values corresponding to the instance of interest
    Input data: Dataset used
    Input data_label: Training dataset labels
    Input std_dev: Standard deviation of the features in the data
    Input LID_x: LID of the instance of interest
    Input data_str: String of the dataset's name
    Input factor: Multiplier of the standard deviation of each feature
    Input N: The number of instances to generate complying with the LID restriction
    Input k: Number of k neighbors to evaluate the LID constraint
    Input freqs: Frequencies of the different values in the dataset for all features
    Input validity: When None, no valid perturbation method called
    Output x_perturbations_array: Array of instances generated from the perturbation
    Output LID_perturbations_array: Array of LID values of the complying generated instances generated from the perturbation
    Output verified_instances: Number of valid instances
    """
    LID_ft = int(LID_x)
    x_perturbations = []
    LID_perturbations = []
    number_instances = 0
    verified_instances = 0
    if LID_ft > len(x):
        LID_ft = len(x)    
    n_permutations = 100
    r_state = 1
    feature_importances = feature_ranking(x,data,data_label,n_permutations,r_state,k,data_str) 
    feature_importance_index = np.argsort(feature_importances)[-int(LID_ft):][::-1]
    while number_instances < N:
        if validity == 'yes':
            x_perturbed = single_feat_perturbation(x,factor,std_dev,data_str,freqs,feature_importance_index)
        else:
            x_perturbed = single_unconstrained_pert(x,factor,std_dev,feature_importance_index)
        LID_x_perturbed, x_perturbed_kNN = LID(x_perturbed,data,k)
        x_perturbations.append(x_perturbed)
        LID_perturbations.append(LID_x_perturbed)
        number_instances += 1
        verified = feature_verification(x_perturbed,data_str)
        if verified:
            verified_instances += 1
    x_perturbations_array = np.array(x_perturbations)
    LID_perturbations_array = np.array(LID_perturbations)
    return x_perturbations_array, LID_perturbations_array, verified_instances

"""
LID-restricted Vanilla Random (LID Van. R.)
"""
def random_perturbator_LID_ver(x,data,std_dev,LID_x,data_str,factor,N,k,epsilon,freqs,validity = None):
    """
    Function used for random generation of new instances
    Input x: Array of numerical values corresponding to the instance of interest
    Input data: Dataset used
    Input std_dev: Standard deviation of the features in the data
    Input LID_x: LID of the instance of interest
    Input data_str: String of the dataset's name
    Input factor: Multiplier of the standard deviation of each feature
    Input N: The number of instances to generate complying with the LID restriction
    Input k: Number of k neighbors to evaluate the LID constraint
    Input epsilon: Allowed distance between the average LID of synthetic neighborhood and the LID of the test instance
    Input freqs: Frequencies of the different values in the dataset for all features
    Input validity: When None, no valid perturbation method called
    Output x_perturbations_array: Array of instances generated from the perturbation
    Output LID_perturbations_array: Array of LID values of the complying generated instances generated from the perturbation
    Output verified_instances: Number of valid instances
    """
    x_perturbations = []
    LID_perturbations = []
    uncompliant = []
    failed = 0
    number_instances = 0
    verified_instances = 0
    while number_instances < N:
        if validity == 'yes':
            x_perturbed = perturbator_function(x,factor,std_dev,data_str,freqs)
        else:
            x_perturbed = np.random.normal(loc=x, scale=factor*std_dev, size=len(std_dev))
        if any((x_perturbed == i).all() for i in uncompliant):
            continue        
        LID_x_perturbed, x_perturbed_kNN = LID(x_perturbed,data,k)
        LID_verified = LID_verification(LID_x_perturbed,epsilon,LID_x)
        if LID_verified:
            x_perturbations.append(x_perturbed)
            LID_perturbations.append(LID_x_perturbed)
            number_instances += 1
            verified = feature_verification(x_perturbed,data_str)
            if verified:
                verified_instances += 1
        else:
            uncompliant.append(x_perturbed)
            failed += 1
        if failed > N/2:
            x_perturbations.append(x_perturbed)
            LID_perturbations.append(LID_x_perturbed)
            number_instances += 1
            failed = 0
            verified = feature_verification(x_perturbed,data_str)
            if verified:
                verified_instances += 1            
    x_perturbations_array = np.array(x_perturbations)
    LID_perturbations_array = np.array(LID_perturbations)
    return x_perturbations_array, LID_perturbations_array, verified_instances

"""
Optimized LID-restricted Vanilla Random (Opt. LID Van. R.)
"""
def iterator_optimizer_ver(x,data,std_dev,LID_x,data_str,k,factor,x_LID_ver_perturbations,LID_ver_perturbations,req_new,freqs,validity = None):
    """
    Function used to look for new random instances that can be used to replace the current ones and improve LID simmilarity
    Input x: Array of numerical values corresponding to the instance of interest
    Input data: Dataset used
    Input std_dev: Standard deviation of the features in the data
    Input LID_x: LID of the instance of interest    
    Input data_str: String of the dataset's name
    Input k: Number of k neighbors to evaluate the LID constraint
    Input factor: Multiplier of the standard deviation of each feature
    Input x_LID_ver_perturbations: Array obtained by the LID_ver_perturbation algorithm
    Input LID_ver_perturbations: LID of the x_perturbations array
    Input req_new: Number of new instances to generate to replace old instances
    Input freqs: Frequencies of the different values in the dataset for all features
    Input validity: When None, no valid perturbation method called
    Output x_perturbations_array: Array of instances generated from the perturbation
    Output LID_perturbations_array: Array of LID values of the complying generated instances generated from the perturbation
    Output verified_instances: Number of valid instances
    Output replacements: Number of instances replaced by the optimization process
    """    
    replacements = 0
    verified_instances = 0
    failed_try = 0
    uncompliant = []
    x_opt_perturbations = x_LID_ver_perturbations.copy()
    LID_opt_perturbations = LID_ver_perturbations.copy()
    while replacements < req_new:
        LID_difference = np.abs(LID_opt_perturbations - LID_x)
        max_LID_difference = np.max(LID_difference)
        max_LID_difference_idx = np.argmax(LID_difference)
        if validity == 'yes':
            x_perturbed = perturbator_function(x,factor,std_dev,data_str,freqs)
        else:
            x_perturbed = np.random.normal(loc=x, scale=factor*std_dev, size=len(std_dev))
        if any((x_perturbed == i).all() for i in uncompliant):
            continue
        LID_x_perturbed, x_perturbed_kNN = LID(x_perturbed,data,k)
        if np.abs(LID_x_perturbed - LID_x) < max_LID_difference:
            x_opt_perturbations[max_LID_difference_idx] = x_perturbed
            LID_opt_perturbations[max_LID_difference_idx] = LID_x_perturbed
            replacements += 1
            x_opt_perturbations = np.array(x_opt_perturbations)
            verified = feature_verification(x_perturbed,data_str)
            if verified:
                verified_instances += 1
        else:
            uncompliant.append(x_perturbed)
            failed_try += 1
        if failed_try > int(req_new*10):
            break 
    return x_opt_perturbations, LID_opt_perturbations, verified_instances, replacements

"""
Feature LID-restricted Vanilla Random (Feat. LID Van. R.)
"""
def feat_perturbator_LID_ver(x,data,std_dev,LID_x,data_str,factor,N,k,epsilon,freqs,validity = None):
    """
    Function used for random generation of new instances
    Input x: Array of numerical values corresponding to the instance of interest
    Input data: Dataset used
    Input std_dev: Standard deviation of the features in the data
    Input LID_x: LID of the instance of interest
    Input data_str: String of the dataset's name
    Input factor: Multiplier of the standard deviation of each feature
    Input N: The number of instances to generate complying with the LID restriction
    Input k: Number of k neighbors to evaluate the LID constraint
    Input epsilon: Allowed distance between the average LID of synthetic neighborhood and the LID of the test instance
    Input freqs: Frequencies of the different values in the dataset for all features
    Input validity: When None, no valid perturbation method called
    Output x_perturbations_array: Array of instances generated from the perturbation
    Output LID_perturbations_array: Array of LID values of the complying generated instances generated from the perturbation
    Output verified_instances: Number of valid instances
    """
    LID_ft = int(LID_x)
    x_perturbations = []
    LID_perturbations = []
    uncompliant = []
    failed = 0
    number_instances = 0
    verified_instances = 0
    if LID_ft > len(x):
        LID_ft = len(x)
    while number_instances < N:
        feature_index = np.random.choice(range(len(std_dev)),size=(LID_ft))
        if validity == 'yes':
            x_perturbed = single_feat_perturbation(x,factor,std_dev,data_str,freqs,feature_index)
        else:
            x_perturbed = single_unconstrained_pert(x,factor,std_dev,feature_index)
        if any((x_perturbed == i).all() for i in uncompliant):
            continue
        LID_x_perturbed, x_perturbed_kNN = LID(x_perturbed,data,k)
        LID_verified = LID_verification(LID_x_perturbed,epsilon,LID_x)
        if LID_verified:
            x_perturbations.append(x_perturbed)
            LID_perturbations.append(LID_x_perturbed)
            number_instances += 1
            verified = feature_verification(x_perturbed,data_str)
            if verified:
                verified_instances += 1
        else:
            uncompliant.append(x_perturbed)
            failed += 1
        if failed > N/2:
            x_perturbations.append(x_perturbed)
            LID_perturbations.append(LID_x_perturbed)
            number_instances += 1
            failed = 0
            verified = feature_verification(x_perturbed,data_str)
            if verified:
                verified_instances += 1            
            print(f'Added a non-compliant neighbor!')
    x_perturbations_array = np.array(x_perturbations)
    LID_perturbations_array = np.array(LID_perturbations)
    return x_perturbations_array, LID_perturbations_array, verified_instances

"""
Feature LID-restricted Vanilla Randomwith Feature Importance (Feat. LID Van. R. FI)
"""
def feat_perturbator_LID_ver_fi(x,data,data_label,std_dev,LID_x,data_str,factor,N,k,epsilon,freqs,validity = None):
    """
    Function used for random generation of new instances
    Input x: Array of numerical values corresponding to the instance of interest
    Input data: Dataset used
    Input data_label: Training dataset labels
    Input std_dev: Standard deviation of the features in the data
    Input LID_x: LID of the instance of interest
    Input data_str: String of the dataset's name
    Input factor: Multiplier of the standard deviation of each feature
    Input N: The number of instances to generate complying with the LID restriction
    Input k: Number of k neighbors to evaluate the LID constraint
    Input epsilon: Allowed distance between the average LID of synthetic neighborhood and the LID of the test instance
    Input freqs: Frequencies of the different values in the dataset for all features
    Input validity: When None, no valid perturbation method called
    Input uncompliant: List of perturbations which do not comply with LID verification
    Output x_perturbations_array: Array of instances generated from the perturbation
    Output LID_perturbations_array: Array of LID values of the complying generated instances generated from the perturbation
    Output verified_instances: Number of valid instances
    """
    LID_ft = int(LID_x)
    x_perturbations = []
    LID_perturbations = []
    uncompliant = []
    failed = 0
    number_instances = 0
    verified_instances = 0
    if LID_ft > len(x):
        LID_ft = len(x)
    n_permutations = 100
    r_state = 1
    feature_importances = feature_ranking(x,data,data_label,n_permutations,r_state,k,data_str)
    feature_index = np.argsort(feature_importances)[-int(LID_ft):][::-1]
    while number_instances < N:
        if validity == 'yes':
            x_perturbed = single_feat_perturbation(x,factor,std_dev,data_str,freqs,feature_index)
        else:
            x_perturbed = single_unconstrained_pert(x,factor,std_dev,feature_index)
        if any((x_perturbed == i).all() for i in uncompliant):
            continue
        LID_x_perturbed, x_perturbed_kNN = LID(x_perturbed,data,k)
        LID_verified = LID_verification(LID_x_perturbed,epsilon,LID_x)
        if LID_verified:
            x_perturbations.append(x_perturbed)
            LID_perturbations.append(LID_x_perturbed)
            number_instances += 1
            verified = feature_verification(x_perturbed,data_str)
            if verified:
                verified_instances += 1
        else:
            uncompliant.append(x_perturbed)
            failed += 1
        if failed > N/2:
            x_perturbations.append(x_perturbed)
            LID_perturbations.append(LID_x_perturbed)
            number_instances += 1
            failed = 0
            verified = feature_verification(x_perturbed,data_str)
            if verified:
                verified_instances += 1            
            print(f'Added a non-compliant neighbor!')        
    x_perturbations_array = np.array(x_perturbations)
    LID_perturbations_array = np.array(LID_perturbations)
    return x_perturbations_array, LID_perturbations_array, verified_instances