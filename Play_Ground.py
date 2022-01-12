#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py


# In[2]:


with h5py.File('data.h5', 'r') as hf:
    X = hf['EEG_values'][:]             #Samples tensor
    y = hf['target_values'][:]          #Targets matrix


# In[3]:


X.shape


# In[4]:


channels_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


# In[95]:


class PCA_MTS():
    Index_Samples             = 0
    Index_length_time_sereis  = 1
    Index_features            = 2
    X_normalized = 0
    
    def __init__(self,X_tensor):
        
        print("Pls make sure that Input tensor X as shape of M: Samples , ni: length of time sereis , m: Numper of features ")        
        self.X_tensor = X_tensor
        
        def global_imports(modulename,shortname = None, asfunction = False):
            if shortname is None: 
                shortname = modulename
            if asfunction is False:
                globals()[shortname] = __import__(modulename)
            else:        
                globals()[shortname] = eval(modulename + "." + shortname) 
                
        global_imports("numpy","np")
        global_imports("pandas","pd")
        global_imports("tensorflow","tf")
        global_imports("tensorflow_probability","tfp")
        global_imports("seaborn","sns")
        global_imports("matplotlib","plt")
        

        
        #mean of each sample per m features cross ni length
        mean_vector_i = tf.divide(tf.reduce_sum(X_tensor, self.Index_length_time_sereis), X_tensor.shape[self.Index_length_time_sereis])
        #This result in tensor of shape (M , m)

        #Now broad-castting the tensor intp (M, ni ,m)
        mean_vector_i = np.tile(mean_vector_i, (1,X_tensor.shape[self.Index_length_time_sereis])).reshape(X_tensor.shape[self.Index_Samples],
                                                                                      X_tensor.shape[self.Index_length_time_sereis],
                                                                                      X_tensor.shape[self.Index_features])

        PCA_MTS.X_normalized  = tf.subtract(X_tensor,mean_vector_i).numpy() #Getting the normalized tensor

        

    def stats_COV(self):
        '''Calculating covariance matrix'''
        
        State_x = PCA_MTS.X_normalized

        DeNormalized_Segma = tfp.stats.covariance(State_x, sample_axis=1, event_axis=2, keepdims=False, name=None)
    
        Segma_COV          = tf.divide(tf.reduce_sum(DeNormalized_Segma, PCA_MTS.Index_Samples), self.X_tensor.shape[PCA_MTS.Index_Samples])
    
        return Segma_COV.numpy() 
    
    def normalized_COV(self):
        '''Calculating correlationg matrix out from covariance matrix'''
        
        store = PCA_MTS.X_normalized
        
        std_x = PCA_MTS.X_normalized
 
    
        std     = tf.math.reduce_std(
            std_x, axis=1, keepdims=False, name=None)
    
        std     = np.tile(std, (1,self.X_tensor.shape[PCA_MTS.Index_length_time_sereis])).reshape(self.X_tensor.shape[PCA_MTS.Index_Samples],
                                                                                      self.X_tensor.shape[PCA_MTS.Index_length_time_sereis],
                                                                                      self.X_tensor.shape[PCA_MTS.Index_features])
        std_x   =  tf.divide(std_x,std)
    
        
        PCA_MTS.X_normalized = std_x
        
        correlation = self.stats_COV()
    
        
        PCA_MTS.X_normalized = store
    
        return correlation
    
    def correlation(self, Columns , figs = (10,10) ,titles = 20):
        
        correlation = self.normalized_COV()
        
        fig, ax = plt.pyplot.subplots(figsize= figs)
        sns.heatmap(tf.math.abs(correlation),
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 12},
                     cmap='coolwarm',                 
                     yticklabels = Columns,
                     xticklabels = Columns,
                     ax = ax)
        plt.pyplot.title('Covariance matrix showing abs correlation coefficients', size = titles)
        plt.pyplot.tight_layout()
        plt.pyplot.show()
        
    
    def eigens(self):
        '''PCA involves projecting the data onto the eigenvectors of the covariance matrix.
        If you don't standardize your data first, these eigenvectors will be all different lengths.
        Then the eigenspace of the covariance matrix will be stretched, leading to similarly "stretched" projections.
        However, there are situations in which you do want to preserve the original variances.
        So I'll save both eigens'''
        
        Corr_eig_vals, Corr_eig_vecs = np.linalg.eig(self.normalized_COV())
        COV_eig_vals, COV_eig_vecs   = np.linalg.eig(self.stats_COV())
        
        return (Corr_eig_vals, Corr_eig_vecs, COV_eig_vals, COV_eig_vecs)
    
    def correlation_explained_var(self,start = 6 , end = 12 , figs = (10,10)):
        '''Percentage of Explained Variance using correlation matrix'''
        eig_vals , eig_vecs , Junk , Junk = self.eigens()
        
        start = start-1
        Explained_Var = [(i / sum(eig_vals))*100 for i in sorted(eig_vals, reverse=True)]
        cum_Explained_Var = np.cumsum(Explained_Var)
        with plt.pyplot.style.context('seaborn-whitegrid'):
            plt.pyplot.figure(figsize=figs)

            plt.pyplot.bar(range(X.shape[self.Index_features]), Explained_Var, alpha=0.5, align='center',
                    label='individual explained variance')
            plt.pyplot.step(range(X.shape[self.Index_features]), cum_Explained_Var, where='mid',
                     label='cumulative explained variance')
            plt.pyplot.ylabel('Explained variance ratio')
            plt.pyplot.xlabel('Principal components')
            plt.pyplot.legend(loc='best')
            plt.pyplot.tight_layout()
            for i in range(start,end,1):
                print('Percentage of Explained Variance using correlation matrix if we used %s Components is:'%(i+1) +str(cum_Explained_Var[i])+'%'+'\n')

    def covariance_explained_var(self,start = 6 , end = 12 , figs = (10,10)):
        '''Percentage of Explained Variance using covariance matrix'''
        Junk , Junk , eig_vals , eig_vecs = self.eigens()
        
        start = start-1
        Explained_Var = [(i / sum(eig_vals))*100 for i in sorted(eig_vals, reverse=True)]
        cum_Explained_Var = np.cumsum(Explained_Var)
        with plt.pyplot.style.context('seaborn-whitegrid'):
            plt.pyplot.figure(figsize=figs)

            plt.pyplot.bar(range(X.shape[self.Index_features]), Explained_Var, alpha=0.5, align='center',
                    label='individual explained variance')
            plt.pyplot.step(range(X.shape[self.Index_features]), cum_Explained_Var, where='mid',
                     label='cumulative explained variance')
            plt.pyplot.ylabel('Explained variance ratio')
            plt.pyplot.xlabel('Principal components')
            plt.pyplot.legend(loc='best')
            plt.pyplot.tight_layout()
            for i in range(start,end,1):
                print('Percentage of Explained Variance using covariance matrix if we used %s Components is:'%(i+1) +str(cum_Explained_Var[i])+'%'+'\n')
 
    def correlation_most_effective_features_pc(self, Columns):
        
        eig_vals , eig_vecs , Junk , Junk = self.eigens()
        
        Dim = np.abs(eig_vecs).shape[0]
        most_important = [np.abs(eig_vecs[i]).argmax() for i in range(Dim)]

        most_important_features = [Columns[most_important[i]] for i in range(Dim)]
        dic = {'PC{}'.format(i):  most_important_features[i] for i in range(Dim)}

        df = pd.DataFrame(dic.items())
        df.columns = ['PC', 'features']
        
        return df

    def covariance_most_effective_features_pc(self, Columns):
        
        Junk , Junk , eig_vals , eig_vecs = self.eigens()
        
        Dim = np.abs(eig_vecs).shape[0]
        most_important = [np.abs(eig_vecs[i]).argmax() for i in range(Dim)]

        most_important_features = [Columns[most_important[i]] for i in range(Dim)]
        dic = {'PC{}'.format(i):  most_important_features[i] for i in range(Dim)}

        df = pd.DataFrame(dic.items())
        df.columns = ['PC', 'features']
        
        return df
    


# In[96]:


Obj = PCA_MTS(X)


# ![image.png](attachment:image.png)

# In[97]:


Obj.stats_COV()


# In[98]:


Obj.correlation(channels_order)


# In[99]:


Obj.correlation_explained_var()


# In[100]:


Obj.covariance_explained_var()


# In[101]:


Obj.correlation_most_effective_features_pc(channels_order)


# In[102]:


Obj.covariance_most_effective_features_pc(channels_order)


# In[103]:


Segma = Obj.stats_COV()


# In[104]:


eig_vals, eig_vecs = np.linalg.eig(Segma)


# In[105]:


S = eig_vecs[:,0:10]


# In[106]:


PC = np.dot(X,S)


# In[107]:


Y = np.dot(PC,S.T) 


# In[108]:


Y.shape


# In[109]:


E_i = Y - X


# In[110]:


E_i = E_i ** 2


# In[111]:


E_i = tf.reduce_sum(E_i, 2)


# In[112]:


E_i = tf.reduce_sum(E_i, 1)


# In[113]:


E_i


# In[114]:


y_test , X_test = y[0:244] , X[0:244]


# In[115]:


y_k , X_k      = y[244:] , X[244:]


# In[325]:


X = ["X" + str(i) for i in range (5)]
y = ["y" + str(i) for i in range (5)]


# In[326]:


X[0] = X_k[np.where(y_k == 0)[0].tolist()]
y[0] = y_k[np.where(y_k == 0)[0].tolist()]


# In[327]:


X[1] = X_k[np.where(y_k == 6.66)[0].tolist()]
y[1] = y_k[np.where(y_k == 6.66)[0].tolist()]


# In[328]:


X[2] = X_k[np.where(y_k == 7.5 )[0].tolist()]
y[2] = y_k[np.where(y_k == 7.5 )[0].tolist()]


# In[329]:


X[3] = X_k[np.where(y_k == 8.57)[0].tolist()]
y[3] = y_k[np.where(y_k == 8.57)[0].tolist()]


# In[330]:


X[4] = X_k[np.where(y_k == 12.0)[0].tolist()]
y[4] = y_k[np.where(y_k == 12.0)[0].tolist()]


# In[320]:


def number_classes(k):
    Obj = ["Obj" + str(i) for i in range (k)]
    return Obj
Obj = number_classes(5)


# In[332]:


S = ["S" + str(i) for i in range (5)]


# In[333]:


E_i = ["E_i" + str(i) for i in range (5)]


# In[341]:


for i in range (5):
    
    Obj[i]       = PCA_MTS(X[i])
    Junk, S[i]   = np.linalg.eig(Obj[i].normalized_COV())
    S[i]         = S[i][:,0:10]
    E_i[i]       = np.dot(np.dot(X[i],S[i]),S[i].T) - X[i]
    E_i[i]       = E_i[i] ** 2
    E_i[i]       = tf.reduce_sum(E_i[i], 2)
    E_i[i]       = tf.reduce_sum(E_i[i], 1)

    


# In[142]:


X_test[0]


# In[208]:


PC= np.dot(X_test[0],S_0)
Y = np.dot(PC,S_0.T) 
E_i = Y - X_test[1]
E_i= E_i ** 2
E_i = tf.reduce_sum(E_i, 1)
E_i = tf.reduce_sum(E_i, 0)


# In[209]:


E_i


# In[210]:


np.dot(PC,S_3.T).shape


# In[211]:


X_test[1].shape


# In[304]:


PC= np.dot(X_test[3],S_1)
Y = np.dot(PC,S_1.T) 


# In[305]:


E_i = Y - X_test[0]


# In[306]:


E_i= E_i ** 2


# In[307]:


tf.reduce_sum(E_i, 1)


# In[308]:


tf.reduce_sum(tf.reduce_sum(E_i, 1), 0)


# In[309]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_0).numpy()) / tf.math.reduce_std(E_i_0).numpy()).numpy()


# In[310]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_1).numpy()) / tf.math.reduce_std(E_i_1).numpy()).numpy()


# In[311]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_2).numpy()) / tf.math.reduce_std(E_i_2).numpy()).numpy()


# In[312]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_3).numpy()) / tf.math.reduce_std(E_i_3).numpy()).numpy()


# In[313]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_4).numpy()) / tf.math.reduce_std(E_i_4).numpy()).numpy()


# In[ ]:


def classification(X_valid):
    class0 = []
    class1 = []
    class3 = []
    class4 = []
    
    PC  = np.dot(X_valid,S_1)
    Y   = np.dot(PC,S_1.T) 
    E_i = Y - X_valid
    E_i = E_i ** 2
    E_i =tf.reduce_sum(tf.reduce_sum(E_i, 1), 0)
    
    for i in range(5):
    
    
    

