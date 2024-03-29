from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.params = None

        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha

    # you need to implement this function

    def fit(self,X,y):
        '''
        This function does not return anything
        
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.__classes = np.unique(y)

        #####equation 5
        N1= 0

        for i in range(0,y.shape[0]):
            if y[i]==1:
                N1+=1
        theta= (N1+ a)/(y.shape[0]+ a + b)
        ##print("/",theta)
        #################
        #print(np.unique(X[0]))
        #####equation 8
        theta1= np.zeros((X.shape[1],1))
        K= np.zeros((18,1))

        for j in range(0,X.shape[1]):
            Attr= X.T[j]
            k= np.unique(Attr).size
            K[j]=k



        ##################

        ######equation 9

        theta2= np.zeros((X.shape[1],1))




        # remove next line and implement from here
        # you are free to use any data structure for paramse
        params1= [theta,X,y,N1,K]
        #params = None

        # do not change the line below
        self.params = params1
    
    # you need to implement this function
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''

        params = self.params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        #remove next line and implement from here
        m= params[0]* np.prod(params[1])
        predictions = np.random.choice(self.__classes,np.unique(Xtest.shape[0]))
        #do not change the line below
        nMatrix1 = np.zeros(Xtest.shape)
        nMatrix2 = np.zeros(Xtest.shape)
        for i in range(0,len(Xtest)):
            for j in range(0,18):
                n1j = 0
                n2j = 0
                for k in range(0,len(params[1])):
                    if(Xtest[i][j] == params[1][k][j] and params[2][k] == 1):
                        n1j += 1
                    if(Xtest[i][j] == params[1][k][j] and params[2][k] == 2):
                        n2j += 1

                theta1j = (n1j + alpha)/(params[3]+(params[4][j]))
                theta2j =  (n2j + alpha)/((len(params[2]) - params[3])+(params[4][j]))
                #temp_tuple = (theta1j, theta2j)
                nMatrix1[i][j] = theta1j
                nMatrix2[i][j] = theta2j
                #inner_list.append(temp_tuple)




        ypred = np.zeros((Xtest.shape[0],1))


        temp1 = 1
        temp2 = 1
        for i in range(0,Xtest.shape[0]):
            temp= nMatrix1[i]
            prod= np.prod(temp)
            a=params[0]*prod
            temp= nMatrix2[i]
            prod= np.prod(temp)
            b=(1-params[0])*prod
            if(a>b):
                ypred[i]=1
            else:
                ypred[i]=2




        predictions= ypred
        return predictions
        
def evaluateBias(y_pred,y_sensitive):
    '''
    This function computes the Disparate Impact in the classification predictions (y_pred),
    with respect to a sensitive feature (y_sensitive).
    
    Inputs:
    y_pred: N length numpy array
    y_sensitive: N length numpy array
    
    Output:
    di (disparateimpact): scalar value
    '''
    #remove next line and implement from here
    di = 0

    count1 = 0
    count2 = 0
    count3 = 0
    count4=  0

    for i in range(0, y_pred.shape[0]):
        if y_sensitive[i]==2:
            count3+=1
            if y_pred[i] == 2:
                count1 += 1

    for i in range(0, y_pred.shape[0]):
        if y_sensitive[i]==1:
            count4+=1
            if y_pred[i] == 2 :
                count2 += 1

    num = count1/count3
    den = count2/count4
    #print(num)
    #print(den)
    di = (num) / (den)
    #do not change the line below
    return di

def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
    Oversamples instances belonging to the sensitive feature value (s != 1)
    
    Inputs:
    X - Data
    y - labels
    s - sensitive attribute
    p - probability of sampling unprivileged customer
    nsamples - size of the resulting data set (2*nsamples)
    
    Output:
    X_sample,y_sample,s_sample
    '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]

    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
