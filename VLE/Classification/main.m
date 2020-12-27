clc
clear
addpath(genpath(pwd));
% dataset_list
datasets = {'CAL500\CAL500','emotions\emotions','medical\medical','llog\llog','enron\enron','image\image','scene\scene','yeast\yeast','slashdot\slashdot','corel5k\corel5k','msra\msra'};
% folder of data
data_folder = 'MLL_Dataset\';
% folder of distribution
distribution_folder = 'distribution\';

for dataN = 1:1
    dataset = strsplit(datasets{dataN},'\');
    dataset = dataset{1};
    data_path = strcat(data_folder, dataset, '\', dataset, '_total_');
    distribution_path = strcat(distribution_folder, dataset, '\', dataset, '_LE_');
    % parameter
    para.tol = 1e-5;    %tolerance during the iteration
    para.epsi = 0.001;  %instances whose distance computed is more than epsi should be penalized
    para.C1 = 1;        %penalty parameter
    para.C2 = 1;        %penalty parameter
    para.ker = 'rbf';   %type of kernel function ('lin', 'poly', 'rbf', 'sam')
    % variable to store measurement results
    dists = zeros(10,5);
    for group = 1:10
       % load distribution [n_sample x n_label]
       distribution_file = strcat(distribution_path, int2str(group),'.mat');
       load(distribution_file);
       train_distributions = train_target';
       % load data
       data_file = strcat(data_path, int2str(group),'.mat');
       load(data_file);
       % preprocessing
       train_features = zscore(train_data); %[n_sample x n_feature]
       train_target = train_distributions;
       train_target = 1 ./ (1 + exp(0 - train_target)); % sigmoid(d_i^j)
       tmp_max = max(train_target,[],2);
       tmp_min = min(train_target,[],2);
       train_target = (train_target-tmp_min) ./ (tmp_max-tmp_min)*2 -1;
       test_target = test_target';
       test_target(find(test_target==0))=-1;
       para.par  = 1*mean(pdist(train_data)); %parameter of kernel function
       % training
       model = amsvr(train_data, train_target, para);
       % predicting
       [label, degree] = predict(test_data, train_data, model);
       % evaluation
       dist = testModel(degree, label, test_target);
       dists(group,:) = dist;
    end
    dist_mean = mean(dists,1);
    dist_std = std(dists,1);
    round(dist_mean,3)
    round(dist_std,3)
end