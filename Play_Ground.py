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


# In[5]:


class PCA_MTS():
    Index_Samples             = 0
    Index_length_time_sereis  = 1
    Index_features            = 2
    X_normalized = 0
    
    def __init__(self,X):
        
        print("Pls make sure that Input tensor X as shape of M: Samples , ni: length of time sereis , m: Numper of features ")        
        self.X = X
        
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
        mean_vector_i = tf.divide(tf.reduce_sum(X, self.Index_length_time_sereis), X.shape[self.Index_length_time_sereis])
        #This result in tensor of shape (M , m)

        #Now broad-castting the tensor intp (M, ni ,m)
        mean_vector_i = np.tile(mean_vector_i, (1,X.shape[self.Index_length_time_sereis])).reshape(X.shape[self.Index_Samples],
                                                                                      X.shape[self.Index_length_time_sereis],
                                                                                      X.shape[self.Index_features])

        PCA_MTS.X_normalized  = tf.subtract(X,mean_vector_i).numpy() #Getting the normalized tensor

        

    def stats_COV(self):
        '''Calculating covariance matrix'''
        
        State_x = PCA_MTS.X_normalized

        DeNormalized_Segma = tfp.stats.covariance(State_x, sample_axis=1, event_axis=2, keepdims=False, name=None)
    
        Segma_COV          = tf.divide(tf.reduce_sum(DeNormalized_Segma, PCA_MTS.Index_Samples), X.shape[PCA_MTS.Index_Samples])
    
        return Segma_COV.numpy() 
    
    def normalized_COV(self):
        '''Calculating correlationg matrix out from covariance matrix'''
        
        store = PCA_MTS.X_normalized
        
        std_x = PCA_MTS.X_normalized
 
    
        std     = tf.math.reduce_std(
            std_x, axis=1, keepdims=False, name=None)
    
        std     = np.tile(std, (1,X.shape[PCA_MTS.Index_length_time_sereis])).reshape(X.shape[PCA_MTS.Index_Samples],
                                                                                      X.shape[PCA_MTS.Index_length_time_sereis],
                                                                                      X.shape[PCA_MTS.Index_features])
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
    


# In[6]:


Obj = PCA_MTS(X)


# ![image.png](attachment:image.png)

# In[7]:


Obj.stats_COV()


# In[8]:


Obj.correlation(channels_order)


# In[9]:


Obj.correlation_explained_var()


# In[10]:


Obj.covariance_explained_var()


# In[11]:


Obj.correlation_most_effective_features_pc(channels_order)


# In[12]:


Obj.covariance_most_effective_features_pc(channels_order)


# In[13]:


Segma = Obj.stats_COV()


# In[14]:


eig_vals, eig_vecs = np.linalg.eig(Segma)


# In[15]:


S = eig_vecs[:,0:10]


# In[16]:


PC = np.dot(X,S)


# In[17]:


Y = np.dot(PC,S.T) 


# In[18]:


Y.shape


# In[19]:


E_i = Y - X


# In[20]:


E_i = E_i ** 2


# In[21]:


E_i = tf.reduce_sum(E_i, 2)


# In[22]:


E_i = tf.reduce_sum(E_i, 1)


# In[23]:


E_i


# In[24]:


y_test , X_test = y[0:244] , X[0:244]


# In[25]:


y_k , X_k      = y[244:] , X[244:]


# In[26]:


X_0 = X_k[np.where(y_k == 0)[0].tolist()]
y_0 = y_k[np.where(y_k == 0)[0].tolist()]


# In[27]:


X_1 = X_k[np.where(y_k == 6.66)[0].tolist()]
y_1 = y_k[np.where(y_k == 6.66)[0].tolist()]


# In[28]:


X_2 = X_k[np.where(y_k == 7.5 )[0].tolist()]
y_2 = y_k[np.where(y_k == 7.5 )[0].tolist()]


# In[29]:


X_3 = X_k[np.where(y_k == 8.57)[0].tolist()]
y_3 = y_k[np.where(y_k == 8.57)[0].tolist()]


# In[30]:


X_4 = X_k[np.where(y_k == 12.0)[0].tolist()]
y_4 = y_k[np.where(y_k == 12.0)[0].tolist()]


# In[31]:


Obj_0   = PCA_MTS(X_0)
Segma_0 = Obj_0.stats_COV()


# In[32]:


Obj_1   = PCA_MTS(X_1)
Segma_1 = Obj_1.stats_COV()


# In[33]:


Obj_2   = PCA_MTS(X_2)
Segma_2 = Obj_2.stats_COV()


# In[34]:


Obj_3   = PCA_MTS(X_3)
Segma_3 = Obj_3.stats_COV()


# In[35]:


Obj_4   = PCA_MTS(X_4)
Segma_4 = Obj_4.stats_COV()


# In[36]:


eig_vals, eig_vecs = np.linalg.eig(Segma_0)
S_0 = eig_vecs[:,0:10]
eig_vals, eig_vecs = np.linalg.eig(Segma_1)
S_1 = eig_vecs[:,0:10]
eig_vals, eig_vecs = np.linalg.eig(Segma_2)
S_2 = eig_vecs[:,0:10]
eig_vals, eig_vecs = np.linalg.eig(Segma_3)
S_3 = eig_vecs[:,0:10]
eig_vals, eig_vecs = np.linalg.eig(Segma_4)
S_4 = eig_vecs[:,0:10]


# In[37]:


PC_0 = np.dot(X_0,S_0)
Y_0 = np.dot(PC_0,S_0.T) 
E_i_0 = Y_0 - X_0
E_i_0 = E_i_0 ** 2
E_i_0 = tf.reduce_sum(E_i_0, 2)
E_i_0 = tf.reduce_sum(E_i_0, 1)


# In[69]:


tf.math.reduce_mean(E_i_0).numpy()


# In[70]:


tf.math.reduce_std(E_i_0).numpy()


# In[38]:


E_i_0


# In[39]:


PC_1 = np.dot(X_1,S_1)
Y_1 = np.dot(PC_1,S_1.T) 
E_i_1 = Y_1 - X_1
E_i_1 = E_i_1 ** 2
E_i_1 = tf.reduce_sum(E_i_1, 2)
E_i_1 = tf.reduce_sum(E_i_1, 1)


# In[40]:


E_i_1


# In[41]:


PC_2 = np.dot(X_2,S_2)
Y_2 = np.dot(PC_2,S_2.T) 
E_i_2 = Y_2 - X_2
E_i_2 = E_i_2 ** 2
E_i_2 = tf.reduce_sum(E_i_2, 2)
E_i_2 = tf.reduce_sum(E_i_2, 1)


# In[42]:


E_i_2


# In[43]:


PC_3 = np.dot(X_3,S_3)
Y_3 = np.dot(PC_3,S_3.T) 
E_i_3 = Y_3 - X_3
E_i_3 = E_i_3 ** 2
E_i_3 = tf.reduce_sum(E_i_3, 2)
E_i_3 = tf.reduce_sum(E_i_3, 1)


# In[44]:


E_i_3


# In[45]:


PC_4 = np.dot(X_4,S_4)
Y_4 = np.dot(PC_4,S_4.T) 
E_i_4 = Y_4 - X_4
E_i_4 = E_i_4 ** 2
E_i_4 = tf.reduce_sum(E_i_4, 2)
E_i_4 = tf.reduce_sum(E_i_4, 1)


# In[46]:


E_i_4


# In[47]:


X_test[0]


# In[48]:


PC= np.dot(X_test[1],S_0)
Y = np.dot(PC,S_0.T) 
E_i = Y - X_test[1]
E_i= E_i ** 2
E_i = tf.reduce_sum(E_i, 1)
E_i = tf.reduce_sum(E_i, 0)


# In[49]:


E_i


# In[50]:


np.dot(PC,S_3.T).shape


# In[51]:


X_test[1].shape


# In[123]:


PC= np.dot(X_test[0],S_2)
Y = np.dot(PC,S_2.T) 


# In[124]:


E_i = Y - X_test[0]


# In[125]:


E_i= E_i ** 2


# In[126]:


tf.reduce_sum(E_i, 1)


# In[127]:


tf.reduce_sum(tf.reduce_sum(E_i, 1), 0)


# In[116]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_0).numpy()) / tf.math.reduce_std(E_i_0).numpy()).numpy()


# In[122]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_1).numpy()) / tf.math.reduce_std(E_i_1).numpy()).numpy()


# In[128]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_2).numpy()) / tf.math.reduce_std(E_i_2).numpy()).numpy()


# In[98]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_3).numpy()) / tf.math.reduce_std(E_i_3).numpy()).numpy()


# In[107]:


(abs(tf.reduce_sum(tf.reduce_sum(E_i, 1), 0) - tf.math.reduce_mean(E_i_4).numpy()) / tf.math.reduce_std(E_i_4).numpy()).numpy()


# In[ ]:




