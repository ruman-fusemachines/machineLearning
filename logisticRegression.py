import numpy as np
from PIL import Image
import glob
import pandas as pd
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt

class LogisticRegression():

    def __init__(self):
        print("Logistic Regression")

    def get_hypothesis(self, X ,W):
        #print(X.shape, W.shape)
        return X.dot(W)

    def get_sigmoid(self,X,W):
        return 1/(1+np.exp(-(self.get_hypothesis(X,W))))

    def get_cost(self,X,W,Y):
        return -( np.sum((Y * np.log(self.get_sigmoid(X,W))) + ((1 - Y)*np.log(1 - (self.get_sigmoid(X,W))))))/len(X)

    def get_gradient(self, X, W, Y):
        # print("Grad",W.shape, X.T.shape, (self.get_hypothesis(X,W) - Y).shape)
        return pd.DataFrame((1/ len(X)) * (X.T.dot( self.get_hypothesis(X, W) -Y) ))

    def load_data(self,filename):

        image_list = []
        label = []
        for filename in glob.glob(filename):
            label = int(filename[-12:-11])
            img = Image.open(filename)
            twoDarray = np.array(img)
            oneDarray = (255.0-twoDarray.ravel())/255.0
            oneDarray = np.append(label, oneDarray)
            image_list.append(oneDarray)

        image_data = np.array(image_list)
        np.random.shuffle(image_data)
        image_data_pd = pd.DataFrame(image_data)
        X = image_data_pd.iloc[:,1:]
        ones = np.ones([len(X), 1])
        X = np.concatenate((ones, X), axis=1)
        Y = pd.DataFrame(image_data_pd.iloc[:,0])
        return   X, Y.values

    def load_weight(self,X):
        return pd.DataFrame(np.zeros(len(X[0,:])))

    def logistic_regression(self, X, W, Y, alpha, max_iteration):
        cost_list =[]
        for i in range(max_iteration):
            W = W - alpha * self.get_gradient(X, W, Y)

            cost = self.get_cost(X,W,Y)

            if i%10 == 0:
                print("Cost:", cost)
                cost_list.append(cost)
        return W,cost, cost_list



filename = 'numerals/*/*.jpg'
fn = LogisticRegression()

X, Y= fn.load_data(filename)


######################################################

X_train,X_rest,Y_train,Y_rest =  train_test_split(X,Y,test_size=0.4)
X_validate,X_test,Y_validate,Y_test = train_test_split(X_rest,Y_rest,test_size=0.5)

# print("Y train", Y_train.shape)
# print("X train", X_train.shape)

weight = fn.load_weight(X) #dataFrame

##################################################

hypo = fn.get_hypothesis(X, weight)

################################################

sigmoid = fn.get_sigmoid(X, weight)

#########################################################

cost = fn.get_cost(X , weight , Y)
# print("Costs",cost)

#####################################################

gradient = fn.get_gradient(X, weight, Y)
# print("Gradient", gradient)


############################################################

newWeight, cost , cost_list= fn.logistic_regression(X, weight, Y, .01, 10000)
x_axis =[]
for i in range(0,10000):
    if i%10 ==0:
        x_axis.append(i)

print(len(x_axis), len(cost_list))
plt.plot(x_axis,cost_list)
plt.show()


# print("lastcost",newWeight,cost)

###############################################################

# weight_list =[]
# cost_list = []
# for i in range(10):
#     W = pd.DataFrame(np.zeros(len(X_train[0,:])))
#     # print(Y_train.shape)
#     # print(X.shape)
#
#     Y_train_one = (Y_train == float(i)).astype(int)
#     # print("Y train", Y_train_one)
#     # print(X.shape, W.shape, Y_train_one.shape)
#     weight,cost = fn.logistic_regression(X_train,W,  Y_train_one, 0.01,300)
#     weight_list.append(weight.T)
#     cost_list.append(cost)
#
# print(weight_list)
# print(cost_list)