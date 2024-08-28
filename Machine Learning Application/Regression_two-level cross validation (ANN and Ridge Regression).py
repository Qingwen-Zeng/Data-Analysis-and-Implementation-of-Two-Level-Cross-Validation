# exercise 8.2.6 fro moste of ann
#
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from dtuimldmtools import draw_neural_net, train_neural_net

# Import
#import importlib_resources
import numpy as np
import sklearn.linear_model as lm
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
    savefig
)
from scipy.io import loadmat
from sklearn import model_selection

from dtuimldmtools import rlr_validate
# Make a list for storing assigned color of learning curve for up to K=10
color_list = [
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:red",
    "tab:blue",
]

#Lode data in aging remove row of missing value
df = pd.read_csv("C:\\Users\\brandon\\Downloads\\Algerian_forest_fires_dataset_UPDATE.csv",skiprows=[0,124,125,126,170], sep=',')
#Change class label to be 0 and 1, and not fire and not fire 
df = pd.get_dummies(df, columns=['Classes']) #make class 1 / k
G = df.drop('Classes_not_fire', axis=1)
#Remove the day
G = G.drop('day', axis=1)
G = G.drop('year', axis=1)
G = G.drop('month', axis=1)

# = G.rename(columns={'Temperature': 'Temp'})
G = G.rename(columns={'Classes_fire': 'Fire'})
print(G.tail())
g = G.drop('Fire', axis=1)

#we make a dataframe for y which is the FWI data, and x which is all the other altibutech which will be usede to pre the FWI

y = g["FWI"].values.reshape(-1, 1)#.squeeze().values.flatten()

X = g.drop('FWI', axis=1)
attributeNames1 = X.columns[:]
attributeNames = [name for name in attributeNames1]
N, M = X.shape
X = X.values

y_stas_true = [] # save the true value of what is testet in the outer loop
#For the inder out K-fold crossvalidation
K = 10 #K_out = K_ind = 10
max_iter = 600
CV_out = model_selection.KFold(K, shuffle=True)
CV_ind = model_selection.KFold(K, shuffle=True)

#_______________________ neural network classifier setup_______________________________
n_hidden_units_list = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]  # number of hidden units list aka h
n_replicates = 1  # number of networks trained in each k-fold
ANN_error_out = np.zeros(K)
y_stats_ANN_est = list()
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
h_list = []
#_______________________ Regretion model setup_______________________________
#add the ofset altribute
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_Offset = np.concatenate((np.ones((X_scaled.shape[0], 1)), X_scaled), 1)
attributeNames_Offset = ["Offset"] + attributeNames
M_Offset = M + 1
#lambdas = np.linspace(10**(-5), 10**2, 11)
lambdas = np.power(10, np.arange(-3.5,2,0.5))
LinearRegression_Waights = np.empty((M_Offset, K))
LinearRegression_error_out = np.empty((K, 1))
y_stats_lin_est = list()
lamda_list = list()

#
w_rlr_ind = np.empty((M_Offset, K))
mu_ind = np.empty((K, M_Offset - 1))
sigma_ind = np.empty((K, M_Offset - 1))

w_rlr = np.empty((M_Offset, K))
mu = np.empty((K, M_Offset - 1))
sigma = np.empty((K, M_Offset - 1))

#______________________ baceline model setup _____________________________________
bl_error_out = np.zeros(K)
y_stats_bl_est = list()
#_________________
# Normalize data for ANN
X = stats.zscore(X)

plt.subplots(1, 1, figsize=(6,6))
plt.subplot(1,1,1)

for k, (train_index_out, test_index_out) in enumerate(CV_out.split(X, y)):
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

    # Extract training and test set for current CV fold (use for inder and linnarregartion)
    X_train_out = X[train_index_out, :]
    y_train_out = y[train_index_out]
    X_test_out = X[test_index_out, :]
    y_test_out = y[test_index_out]

    X_train_out_offset = X_Offset[train_index_out, :]
    X_test_out_offset = X_Offset[test_index_out, :]


    # Extract training and test set for current CV fold, convert to tensors for ann model
    X_train_out_Tensor = torch.Tensor(X[train_index_out, :])
    y_train_out_Tensor = torch.Tensor(y[train_index_out])
    X_test_out_Tensor = torch.Tensor(X[test_index_out, :])
    y_test_out_Tensor = torch.Tensor(y[test_index_out])

    loop_lamda_list = []
    ANN_error = np.zeros(len(n_hidden_units_list))
    LinearRegression_error = np.zeros(K)
    for b, (train_index_ind, test_index_ind) in enumerate(CV_out.split(X_train_out, y_train_out)):
        #print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))
        
        # Extract training and test set for current CV fold, liner regrtion model ann model
        y_train_ind = y_train_out[train_index_ind]
        y_test_ind = y_train_out[test_index_ind]
        X_train_ind_offset = X_train_out_offset[train_index_ind, :]
        X_test_ind_offset = X_train_out_offset[test_index_ind, :]

        # Extract training and test set for current CV fold, convert to tensors for ann model
        X_train_ind_Tensor = torch.Tensor(X_train_out[train_index_ind, :])
        y_train_ind_Tensor = torch.Tensor(y_train_out[train_index_ind])
        X_test_ind_Tensor = torch.Tensor(X_train_out[test_index_ind, :])
        y_test_ind_Tensor = torch.Tensor(y_train_out[test_index_ind])

        #________________________________ANN model______________________________________________

        for i in range(len(n_hidden_units_list)):
            print("outer loop:",k,"inner loop:",b,"h index:",i)
            n_hidden_units = int(n_hidden_units_list[i])
            # Define the model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            ) 
            # Train the net on training data ( the ind tenceror data)
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train_ind_Tensor,
                y=y_train_ind_Tensor,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )


            # Determine estimated class labels for test set
            y_test_est_ann = net(X_test_ind_Tensor)

            # Determine errors and errors
            se = (y_test_est_ann.float() - y_test_ind_Tensor.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test_ind_Tensor)) #.data.numpy()  # mean
            ANN_error[i] += mse #errors.append(mse)  # store error rate for current CV fold

        #______________________________Liner regration_________________________"
        internal_cross_validation = 10

        (
            opt_val_err,
            opt_lambda,
            mean_w_vs_lambda,
            train_err_vs_lambda,
            test_err_vs_lambda,
        ) = rlr_validate(X_train_ind_offset, y_train_ind, lambdas, internal_cross_validation)

        # Standardize inder fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        mu_ind[b, :] = np.mean(X_train_ind_offset[:, 1:], 0)
        sigma_ind[b, :] = np.std(X_train_ind_offset[:, 1:], 0)

        X_train_ind_offset[:, 1:] = (X_train_ind_offset[:, 1:] - mu_ind[b, :]) / sigma_ind[b, :]
        X_test_ind_offset[:, 1:] = (X_test_ind_offset[:, 1:] - mu_ind[b, :]) / sigma_ind[b, :]

        Xty = X_train_ind_offset.T @ y_train_ind
        XtX = X_train_ind_offset.T @ X_train_ind_offset

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M_Offset)
        loop_lamda_list.append(opt_lambda)

        
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        w_rlr_ind[:, b] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()


        LinearRegression_error[b] = (
            np.square(y_test_ind - (X_test_ind_offset @ w_rlr_ind[:, k]).reshape(-1, 1)).sum(axis=0) / y_test_ind.shape[0]
        )
                

    #________________________________ Baceline model ___________________________________________
    print("out")
    print("Baceline model")
    bl_list = list() 
    for Q in range(len(y_test_out)):
        bl_list.append(np.mean(y_train_out))
    
    y_test_est_bl_out = bl_list
   
    
    cv_mean = np.mean(y_train_out)
    bl_error_out[k] =  np.square(y_test_out-cv_mean).sum()/y_test_out.shape[0] #errors.append(mse)  # store error rate for current CV fold
    #________________________________lingnar regretuion out ____________________________________
    print("LinearRegression")

    loop_lamda_list = np.array(loop_lamda_list)
    lam = loop_lamda_list[np.argmin(LinearRegression_error)]
    internal_cross_validation = 10
    lam_list = np.array([lam])

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train_out_offset, y_train_out, lam_list, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train_out_offset[:, 1:], 0)
    sigma[k, :] = np.std(X_train_out_offset[:, 1:], 0)

    X_train_out_offset[:, 1:] = (X_train_out_offset[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test_out_offset[:, 1:] = (X_test_out_offset[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train_out_offset.T @ y_train_out
    XtX = X_train_out_offset.T @ X_train_out_offset

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M_Offset)
    lamda_list.append(opt_lambda)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()


    y_test_est_LinearRegression_out = (X_test_out_offset @ w_rlr[:, k]).reshape(-1, 1)
    LinearRegression_error_out[k] = (
        np.square(y_test_out - y_test_est_LinearRegression_out).sum(axis=0) / y_test_out.shape[0]
    )
    
    #________________________________ANN model out______________________________________________
    
    print("ANN")
    #fist find the best ANN model 
    #ANN_select[k] = n_hidden_units_list[np.argmin(ANN_error)] # get the best h in the fold
    ny_h = int(n_hidden_units_list[np.argmin(ANN_error)])
    h_list.append(ny_h)
    model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, ny_h),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(ny_h, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
    )

    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train_out_Tensor,
        y=y_train_out_Tensor,
        n_replicates=n_replicates,
        max_iter=max_iter,
            )

    # Determine estimated class labels for test set
    y_test_est_ann_out = net(X_test_out_Tensor)

    # Determine errors and errors
    se = (y_test_est_ann_out.float() - y_test_out_Tensor.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test_out_Tensor)) #.data.numpy()  # mean
    ANN_error_out[k] += mse #errors.append(mse)  # store error rate for current CV fold

    
    # Display the learning curve for the best net in the current fold
    
    plt.plot(learning_curve, color=color_list[k])
    plt.xlabel("Iterations")
    plt.xlim((0, max_iter))
    plt.ylabel("Loss")
    plt.title("ANN Optimization Trace - Regression")
    

    #__________ gem værdiger af  ____________
    y_stats_bl_est.extend(np.array(y_test_est_bl_out))
    y_stats_lin_est.extend(y_test_est_LinearRegression_out.flatten())
    y_stats_ANN_est.extend(y_test_est_ann_out.detach().numpy().flatten())
    y_stas_true.append(y_test_out)

plt.grid()
plt.legend(['CV Fold 1', 'CV Fold 2', 'CV Fold 3', 'CV Fold 4', 'CV Fold 5', 'CV Fold 6', 'CV Fold 7', 'CV Fold 8', 'CV Fold 9', 'CV Fold 10'])
plt.savefig('optimization_trace_regression.pdf', bbox_inches='tight')
plt.show()


#Display

# Display the MSE across folds
#summaries_axes[1].bar(
#    np.arange(1, K + 1), np.squeeze(np.asarray(ANN_error_out)), color=color_list
#)
#summaries_axes[1].set_xlabel("Fold")
#summaries_axes[1].set_xticks(np.arange(1, K + 1))
#summaries_axes[1].set_ylabel("MSE")
#summaries_axes[1].set_title("Test mean-squared-error ANN")

print("Diagram of best neural net in last fold:")
weights = [net[i].weight.data.numpy().T for i in [0, 2]]
biases = [net[i].bias.data.numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print(
    "\nEstimated generalization error for ANN, RMSE: {0}".format(
        round(np.sqrt(np.mean(ANN_error_out)), 4)
    )
)

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of
# the true/known value - these values should all be along a straight line "y=x",
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value

y_stas_true = np.concatenate(y_stas_true).flatten()
### MSE plots

# ANN

plt.subplots(1, 3, figsize=(12,6))
plt.subplot(1,3,1)
y_est = np.array(y_stats_ANN_est)
y_true = np.array(y_stas_true)
axis_range = [np.min([y_est, y_true]) - 1, np.max([y_est, y_true]) + 1]
plt.plot(axis_range, axis_range, "k--")
plt.plot(y_true, y_est, "ob", alpha=0.25)
plt.legend(["Perfect estimation", "Model estimations"])
plt.title("ANN")
plt.ylim(axis_range)
plt.xlim(axis_range)
plt.xlabel("True value")
plt.ylabel("Estimated value")
plt.grid()
#plt.savefig('ann_est_true.pdf', bbox_inches='tight')
#plt.show()


## Linear regression

plt.subplot(1,3,2)
y_est = np.array(y_stats_lin_est)
y_true = np.array(y_stas_true)
axis_range = [np.min([y_est, y_true]) - 1, np.max([y_est, y_true]) + 1]
plt.plot(axis_range, axis_range, "k--")
plt.plot(y_true, y_est, "ob", alpha=0.25)
plt.legend(["Perfect estimation", "Model estimations"])
plt.title("RLR")
plt.ylim(axis_range)
plt.xlim(axis_range)
plt.xlabel("True value")
plt.ylabel("Estimated value")
plt.grid()
#plt.savefig('rlr_est_true.pdf', bbox_inches='tight')
#plt.show()



# #Baceline
plt.subplot(1,3,3)
y_est = np.array(y_stats_bl_est)
y_true = np.array(y_stas_true)
axis_range = [np.min([y_est, y_true]) - 1, np.max([y_est, y_true]) + 1]
plt.plot(axis_range, axis_range, "k--")
plt.plot(y_true, y_est, "ob", alpha=0.25)
plt.legend(["Perfect estimation", "Model estimations"])
plt.title("Baseline")
plt.ylim(axis_range)
plt.xlim(axis_range)
plt.xlabel("True value")
plt.ylabel("Estimated value")
plt.grid()
plt.tight_layout(pad=2)
plt.savefig('comparison_est_true.pdf', bbox_inches='tight')
plt.show()

# Error plot bl, ANN and ling reg
plt.subplots(1, 2, figsize=(12, 4))
plt.subplot(1,2,1)
plt.grid(zorder=0)

first = np.squeeze(np.asarray(ANN_error_out))
second = np.squeeze(np.asarray(LinearRegression_error_out))
third = np.squeeze(np.asarray(bl_error_out))
labels = [i for i in range(1,11)]
x = np.arange(len(labels))
width = 0.25 
plt.bar(x - width, first, width, label = 'ANN', zorder=2)
plt.bar(x, second, width, label='RLR', zorder=2)
plt.bar(x + width, third, width, label='Baseline', zorder=2)
plt.ylabel('MSE')
plt.xlabel('Fold')
#plt.title('The minimum error of each fold of 3 models')
plt.xticks(x, labels=labels)
plt.legend()

plt.subplot(1,2,2)
plt.grid(zorder=0)

first = np.squeeze(np.asarray(ANN_error_out))
second = np.squeeze(np.asarray(LinearRegression_error_out))
#third = np.squeeze(np.asarray(bl_error_out))
labels = [i for i in range(1,11)]
x = np.arange(len(labels))
width = 0.4 
plt.bar(x - width/2, first, width, label = 'ANN', zorder=2)
plt.bar(x + width/2, second, width, label='RLR', zorder=2)
#plt.bar(x + width, third, width, label='Baseline', color='green')
plt.ylabel('MSE')
plt.xlabel('Fold')
#plt.title('The minimum error of each fold of 3 models')
plt.xticks(x, labels=labels)
plt.legend()

plt.tight_layout(pad=2)
plt.savefig('comparison_test_MSE.pdf', bbox_inches='tight')
plt.show()




# plt.bar(
#     np.arange(1, K + 1), np.squeeze(np.asarray(LinearRegression_error_out)), color=color_list, zorder=3
# )
# plt.xlabel("Fold")
# plt.xticks(np.arange(1, K + 1))
# plt.ylabel("MSE")
# plt.title("Test MSE - RLR")

# plt.subplot(1,3,2)
# plt.grid(zorder=0)
# plt.bar(
#     np.arange(1, K + 1), np.squeeze(np.asarray(ANN_error_out)), color=color_list, zorder=3
# )
# plt.xlabel("Fold")
# plt.xticks(np.arange(1, K + 1))
# plt.ylabel("MSE")
# plt.title("Test MSE - ANN")

# plt.subplot(1,3,3)
# plt.grid(zorder=0)
# plt.bar(
#     np.arange(1, K + 1), np.squeeze(np.asarray(bl_error_out)), color=color_list, zorder=3
# )
# plt.xlabel("Fold")
# plt.xticks(np.arange(1, K + 1))
# plt.ylabel("MSE")
# plt.title("Test MSE - Baseline")

# plt.tight_layout(pad=2)
# #plt.savefig('comparison_test_MSE.pdf', bbox_inches='tight')
# #plt.show()

#--------------------------------- print info ------------------------

#ANN
print("ANN")
print(h_list)
print(ANN_error_out)
#Linar
print("LinarRegartion")
print(lamda_list)
print(LinearRegression_error_out)
#BL
print("Baceline error")
print(bl_error_out)

#________________________ Stas 

# perform statistical comparison of the models
# compute z with squared error.

import scipy.stats
import scipy.stats as st 

#ANN
zA = np.abs(y_stas_true - y_stats_ANN_est) ** 2
zB = np.abs(y_stas_true - y_stats_lin_est) ** 2
zC = np.abs(y_stas_true - y_stats_bl_est) ** 2
print('compute confidence interval of models')
print("ANN")
alpha = 0.05
CIA = st.t.interval(
    1 - alpha, df=len(zA) - 1, loc=np.mean(zA), scale=st.sem(zA)
)  # Confidence interval
print(CIA)


print("lin")
alpha = 0.05
CIA = st.t.interval(
    1 - alpha, df=len(zB) - 1, loc=np.mean(zB), scale=st.sem(zB)
)  # Confidence interval
print(CIA)

print("bl")
alpha = 0.05
CIA = st.t.interval(
    1 - alpha, df=len(zC) - 1, loc=np.mean(zC), scale=st.sem(zC)
)  # Confidence interval
print(CIA)

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
print("ann og lin")
z = zA - zB
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(CI)
print(p)

print("ann og bl")
z = zA - zC
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(CI)
print(p)

print("lin og bl")
z = zB - zC
CI = st.t.interval(
    1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z)
)  # Confidence interval
p = 2 * st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(CI)
print(p)

