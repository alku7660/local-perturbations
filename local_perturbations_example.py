"""
Example collecting numerical and binary explanations for proposed local perturbations algorithms
"""

"""
Imports
"""
from local_perturbations import *
from true_explanations_search import *

#Parameters
seed_int = 10
np.random.seed(seed_int)
data_str = 'synth4'
train_fraction = 0.8
factor = 0.1
N = 500 # Number of neighbors to generate: 500
k = 50 # Number of neighbors to gather in kNN (recommended to be higher than minimum number of features)
number_n = int(N*0.02) #Number of minimum instances to extract from the training dataset if no opposite class is found (2% of 500 is 10)
req_new = 50 # Number of neighbors to replace current neighbors due to quality: 50
number_n_opt = int(N*0.01) #Number of minimum instances to extract from the training dataset if no opposite class is found during optimizer step (1% of 500 is 5)
penal = 'l2'
validity = 'yes' # Change to anything else to allow unconstrained perturbations (may obtain implausible feature-valued instances) 

train_data, train_data_target, test_data, test_data_target = load_dataset(data_str,train_fraction)
normal_train_data, train_limits = normalization_train(train_data,data_str)
normal_test_data = normalization_test(test_data,train_limits)
epsilon = 0.5*normal_train_data.shape[1]

#Fidelity ratio initialization (When calculating the total fidelity ratio of the whole test dataset)
fid_feat_random_vanilla_log_reg = 0
fid_feat_random_vanilla_dt = 0
fid_feat_random_vanilla_fi_log_reg = 0
fid_feat_random_vanilla_fi_dt = 0
fid_LID_ver_random_log_reg = 0
fid_LID_ver_random_dt = 0
fid_opt_random_log_reg = 0
fid_opt_random_dt = 0
fid_feat_LID_ver_log_reg = 0
fid_feat_LID_ver_dt = 0
fid_feat_LID_ver_fi_log_reg = 0
fid_feat_LID_ver_fi_dt = 0


#Global model fit
lr = LogisticRegression()
log_reg_model = lr.fit(normal_train_data,train_data_target)
dt_model = DecisionTreeClassifier()
dt_model = dt_model.fit(normal_train_data,train_data_target)

#Instance selection
random_idx = np.random.randint(0,len(normal_test_data))
x = test_data[random_idx]
x_normalized = normal_test_data[random_idx]
x_global_log_reg_prediction = log_reg_model.predict(x_normalized.reshape(1, -1))
x_global_dt_prediction = dt_model.predict(x_normalized.reshape(1, -1))

#Specific parameters for the synth4 dataset (may need to be adjusted for a given dataset)
if data_str == 'synth4':
    closest_x = minimize_closest_point_f3(x)
    true_exp_x = expf_3(closest_x)
    index_feat_values = None # List indicating the indices of categorical features and the corresponding different values each feature may have as tuples. Example: dataset with 4 binary features, use: index_feat_values = [(0,[0,1]),(1,[0,1]),(2,[0,1]),(3,[0,1])] 
    freqs_feat = None # Dict indicating the probability distribution of the different features. Use freqs_feat = find_freqs_laplace(index_feat_values,normal_train_data) when dealing with other datasets. Check the function @ support_local_perturbations.py

LID_x, x_kNN = LID(x_normalized,normal_train_data,k)
std_dev = np.std(normal_train_data, axis=0)

 # Random perturbations with only int(LID) number of features considered
print(f'--- Started Feature Random Vanilla perturbation generation ---')
x_feat_random_vanilla_pert, LID_feat_random_vanilla_pert, valid_feat_random_vanilla_pert = feat_perturbator_vanilla(x_normalized,normal_train_data,std_dev,LID_x,data_str,factor,N,k,freqs_feat,validity)
x_feat_random_vanilla_pert_log_reg_target = log_reg_model.predict(x_feat_random_vanilla_pert)
x_feat_random_vanilla_pert_dt_target = dt_model.predict(x_feat_random_vanilla_pert)
x_feat_random_vanilla_pert, x_feat_random_vanilla_pert_log_reg_target, x_feat_random_vanilla_pert_dt_target, found_feat_vanilla = add_neighbors(x_normalized,x_feat_random_vanilla_pert,x_feat_random_vanilla_pert_dt_target,
                                                                                                                                                x_feat_random_vanilla_pert_log_reg_target,number_n,normal_train_data,train_data_target,data_str)
valid_feat_random_vanilla_pert += found_feat_vanilla
feat_random_vanilla_exp_num, feat_random_vanilla_exp_bin, fid_feat_random_vanilla_log_reg, fid_feat_random_vanilla_dt = local_model_application(x_normalized,x_feat_random_vanilla_pert,x_feat_random_vanilla_pert_log_reg_target, 
                                                                                                                                                x_feat_random_vanilla_pert_dt_target,fid_feat_random_vanilla_log_reg,fid_feat_random_vanilla_dt,
                                                                                                                                                x_global_log_reg_prediction,x_global_dt_prediction,data_str,int(epsilon),penal)

print(f'--- Started Feature Random Vanilla with Feature Importance perturbation generation ---')
x_feat_random_vanilla_fi_pert, LID_feat_random_vanilla_fi_pert, valid_feat_random_vanilla_fi_pert = feat_perturbator_vanilla_fi(x_normalized,normal_train_data,train_data_target,std_dev,LID_x,data_str,factor,N,k,freqs_feat,validity)
x_feat_random_vanilla_fi_pert_log_reg_target = log_reg_model.predict(x_feat_random_vanilla_fi_pert)
x_feat_random_vanilla_fi_pert_dt_target = dt_model.predict(x_feat_random_vanilla_fi_pert)
x_feat_random_vanilla_fi_pert, x_feat_random_vanilla_fi_pert_log_reg_target, x_feat_random_vanilla_fi_pert_dt_target, found_feat_vanilla_fi = add_neighbors(x_normalized,x_feat_random_vanilla_fi_pert,x_feat_random_vanilla_fi_pert_dt_target,
                                                                                                                                                            x_feat_random_vanilla_fi_pert_log_reg_target,number_n,normal_train_data,train_data_target,data_str)
valid_feat_random_vanilla_fi_pert += found_feat_vanilla_fi
feat_random_vanilla_fi_exp_num, feat_random_vanilla_fi_exp_bin, fid_feat_random_vanilla_fi_log_reg, fid_feat_random_vanilla_fi_dt = local_model_application(x_normalized,x_feat_random_vanilla_fi_pert,x_feat_random_vanilla_fi_pert_log_reg_target,
                                                                                                                                                            x_feat_random_vanilla_fi_pert_dt_target,fid_feat_random_vanilla_fi_log_reg,fid_feat_random_vanilla_fi_dt,
                                                                                                                                                            x_global_log_reg_prediction,x_global_dt_prediction,data_str,int(epsilon),penal)

print(f'--- Started LID ver. Random perturbation generation ---')
x_LID_ver_random_pert, LID_ver_random_pert, valid_pert_random_LID_ver = random_perturbator_LID_ver(x_normalized,normal_train_data,std_dev,LID_x,data_str,factor,N,k,epsilon,freqs_feat,validity)
x_LID_ver_random_pert_log_reg_target = log_reg_model.predict(x_LID_ver_random_pert)
x_LID_ver_random_pert_dt_target = dt_model.predict(x_LID_ver_random_pert)
x_LID_ver_random_pert, x_LID_ver_random_pert_log_reg_target, x_LID_ver_random_pert_dt_target, found_LID_ver = add_neighbors(x_normalized,x_LID_ver_random_pert,x_LID_ver_random_pert_dt_target,
                                                                                                                            x_LID_ver_random_pert_log_reg_target,number_n,normal_train_data,train_data_target,data_str)
valid_pert_random_LID_ver += found_LID_ver
LID_ver_random_local_exp_num, LID_ver_random_local_exp_bin, fid_LID_ver_random_log_reg, fid_LID_ver_random_dt = local_model_application(x_normalized,x_LID_ver_random_pert,x_LID_ver_random_pert_log_reg_target,x_LID_ver_random_pert_dt_target,
                                                                                                                                        fid_LID_ver_random_log_reg,fid_LID_ver_random_dt,x_global_log_reg_prediction,x_global_dt_prediction,data_str,
                                                                                                                                        int(epsilon),penal)

print(f'--- Started Random perturbation optimization ---')
x_opt_random_pert, LID_opt_pert, valid_pert_random_opt, opt_replacements = iterator_optimizer_ver(x_normalized,normal_train_data,std_dev,LID_x,data_str,k,factor,x_LID_ver_random_pert,LID_ver_random_pert,req_new,freqs_feat,validity)
x_opt_random_pert_log_reg_target = log_reg_model.predict(x_opt_random_pert)
x_opt_random_pert_dt_target = dt_model.predict(x_opt_random_pert)
x_opt_random_pert, x_opt_random_pert_log_reg_target, x_opt_random_pert_dt_target, found_opt = add_neighbors(x_normalized,x_opt_random_pert,x_opt_random_pert_dt_target,x_opt_random_pert_log_reg_target,
                                                                                                            number_n_opt,normal_train_data,train_data_target,data_str)
valid_pert_random_opt += found_opt
opt_replacements += found_opt
opt_random_local_exp_num, opt_random_local_exp_bin, fid_opt_random_log_reg, fid_opt_random_dt = local_model_application(x_normalized,x_opt_random_pert,x_opt_random_pert_log_reg_target,
                                                                                                                        x_opt_random_pert_dt_target,fid_opt_random_log_reg,fid_opt_random_dt,
                                                                                                                        x_global_log_reg_prediction,x_global_dt_prediction,data_str,int(epsilon),penal)

print(f'--- Started Feature LID verified perturbation generation ---')
x_feat_LID_ver_pert, LID_feat_LID_ver_pert, valid_feat_LID_ver_pert = feat_perturbator_LID_ver(x_normalized,normal_train_data,std_dev,LID_x,data_str,factor,N,k,epsilon,freqs_feat,validity)
x_feat_LID_ver_pert_log_reg_target = log_reg_model.predict(x_feat_LID_ver_pert)
x_feat_LID_ver_pert_dt_target = dt_model.predict(x_feat_LID_ver_pert)
x_feat_LID_ver_pert, x_feat_LID_ver_pert_log_reg_target, x_feat_LID_ver_pert_dt_target, found_feat_LID_ver = add_neighbors(x_normalized,x_feat_LID_ver_pert,x_feat_LID_ver_pert_dt_target,x_feat_LID_ver_pert_log_reg_target,
                                                                                                                            number_n,normal_train_data,train_data_target,data_str)
valid_feat_LID_ver_pert += found_feat_LID_ver
feat_LID_ver_exp_num, feat_LID_ver_exp_bin, fid_feat_LID_ver_log_reg, fid_feat_LID_ver_dt = local_model_application(x_normalized,x_feat_LID_ver_pert,x_feat_LID_ver_pert_log_reg_target,x_feat_LID_ver_pert_dt_target,
                                                                                                                    fid_feat_LID_ver_log_reg,fid_feat_LID_ver_dt,x_global_log_reg_prediction,x_global_dt_prediction,
                                                                                                                    data_str,int(epsilon),penal)

print(f'--- Started Feature LID verified with Feature Importance perturbation generation ---')
x_feat_LID_ver_fi_pert, LID_feat_LID_ver_fi_pert, valid_feat_LID_ver_fi_pert = feat_perturbator_LID_ver_fi(x_normalized,normal_train_data,train_data_target,std_dev,LID_x,data_str,factor,N,k,epsilon,freqs_feat,validity)
x_feat_LID_ver_fi_pert_log_reg_target = log_reg_model.predict(x_feat_LID_ver_fi_pert)
x_feat_LID_ver_fi_pert_dt_target = dt_model.predict(x_feat_LID_ver_fi_pert)
x_feat_LID_ver_fi_pert, x_feat_LID_ver_fi_pert_log_reg_target, x_feat_LID_ver_fi_pert_dt_target, found_feat_LID_ver_fi = add_neighbors(x_normalized,x_feat_LID_ver_fi_pert,x_feat_LID_ver_fi_pert_dt_target,
                                                                                                                                        x_feat_LID_ver_fi_pert_log_reg_target,number_n,normal_train_data,train_data_target,data_str)
valid_feat_LID_ver_fi_pert += found_feat_LID_ver_fi
feat_LID_ver_fi_exp_num, feat_LID_ver_fi_exp_bin, fid_feat_LID_ver_fi_log_reg, fid_feat_LID_ver_fi_dt = local_model_application(x_normalized,x_feat_LID_ver_fi_pert,x_feat_LID_ver_fi_pert_log_reg_target,x_feat_LID_ver_fi_pert_dt_target,
                                                                                                                                fid_feat_LID_ver_fi_log_reg,fid_feat_LID_ver_fi_dt,x_global_log_reg_prediction,x_global_dt_prediction,data_str,
                                                                                                                                int(epsilon),penal)

percentage_valid_pert_random_LID_ver = (valid_pert_random_LID_ver/len(x_LID_ver_random_pert))*100
percentage_valid_pert_random_opt = (valid_pert_random_opt/opt_replacements)*100
percentage_valid_pert_feat_random_vanilla = (valid_feat_random_vanilla_pert/len(x_feat_random_vanilla_pert))*100
percentage_valid_pert_feat_random_vanilla_fi = (valid_feat_random_vanilla_fi_pert/len(x_feat_random_vanilla_fi_pert))*100
percentage_valid_pert_feat_LID_ver = (valid_feat_LID_ver_pert/len(x_feat_LID_ver_pert))*100
percentage_valid_pert_feat_LID_ver_fi = (valid_feat_LID_ver_fi_pert/len(x_feat_LID_ver_fi_pert))*100

print(f'---------------------------------------------------------------------------------')
print(f'---------------------------- Numerical Explanations: ----------------------------')
print(f'    Ground truth explanation      :{true_exp_x}                          ')
print(f'    Feat. Van. R. explanation     :{feat_random_vanilla_exp_num}         ')
print(f'    Feat. Van. R. FI explanation  :{feat_random_vanilla_fi_exp_num}      ')
print(f'    LID Van. R. explanation       :{LID_ver_random_local_exp_num}        ')
print(f'    Opt. LID Van. R. explanation  :{opt_random_local_exp_num}        ')
print(f'   Feat. LID Van. R. explanation  :{feat_LID_ver_exp_num}        ')
print(f' Feat. LID Van. R. FI explanation :{feat_LID_ver_fi_exp_num}        ')
print(f'---------------------------------------------------------------------------------')

print(f'---------------------------------------------------------------------------------')
print(f'------------------------------ Binary Explanations: -----------------------------')
print(f'    Ground truth explanation      :{true_exp_x}                          ')
print(f'    Feat. Van. R. explanation     :{feat_random_vanilla_exp_bin}         ')
print(f'    Feat. Van. R. FI explanation  :{feat_random_vanilla_fi_exp_bin}      ')
print(f'    LID Van. R. explanation       :{LID_ver_random_local_exp_bin}        ')
print(f'    Opt. LID Van. R. explanation  :{opt_random_local_exp_bin}        ')
print(f'   Feat. LID Van. R. explanation  :{feat_LID_ver_exp_bin}        ')
print(f' Feat. LID Van. R. FI explanation :{feat_LID_ver_fi_exp_bin}        ')
print(f'---------------------------------------------------------------------------------')