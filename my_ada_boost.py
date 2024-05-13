import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)  

class My_AdaBoost:
    def __init__(self, ensemble_size=5):
        self.ensemble_size = ensemble_size
        self.estimators = []
        self.model_weights = []

    def learn(self, X, y, base_alg=[GaussianNB()], lr=1):
        if type(base_alg) != list:
            raise TypeError('The algorithms should be added in the form of a list! Use square brackets []!')
                
        n_row, n_col = X.shape
        alphas = pd.Series(np.array([1/n_row]*n_row), index=X.index)
        alg=None
        
        for n in range(self.ensemble_size):

            if len(base_alg)==1:
                alg=base_alg[0]
            else:
                alg = np.random.choice(base_alg)
            
            print('Model used:',n+1, alg)
            model = alg.fit(X, y, sample_weight=alphas)
            y_pred = model.predict(X)
            
            error = (y_pred!=y).astype(int)
            
            weighted_error=(alphas*error).sum()
            model_weight =1/2 * np.log((1 - weighted_error+ 1e-20) / (weighted_error + 1e-20))#zbog deljenja 0om, gde se dobija Nan
            self.model_weights.append(model_weight)
            self.estimators.append(model)
            
            factor=np.exp(-lr*model_weight*y_pred*y)#ovde dobijamo nove alphe
            alphas*=factor
            alphas/=np.sum(alphas)
            
        print(f"\nModelf weights: {self.model_weights}")

        self.model_evaluation(X, y)
            

    def predict(self, X):
        
        predictions=pd.DataFrame()
        k=1
        for model in self.estimators:
            predictions[f'{k} model']=(model.predict(X)).T
            k+=1
        
        conf_class_1= np.abs(predictions.replace(-1,0).dot(self.model_weights)/sum(self.model_weights))
        conf_class_neg_1= np.abs(predictions.replace(1,0).dot(self.model_weights)/sum(self.model_weights))
        
        predictions['final_prediction']=np.sign(predictions.dot(self.model_weights))
        predictions['conf_class_1']=conf_class_1
        predictions['conf_class_neg_1']=conf_class_neg_1
        
        print(predictions)
        

        return predictions
    
    def model_evaluation(self, X, y):
        predictions=pd.DataFrame()
        k=1
        for model in self.estimators:
            predictions[f'{k} model']=(model.predict(X)).T
            k+=1
        predictions['prediction']=np.sign(predictions.dot(self.model_weights))
        
        print('\nConfidence of each model in ansambl:\n')
        print((predictions.iloc[:, :-1].add(y, axis=0).abs()/2).mean())
        print()
    
    
data = pd.read_csv('data/drugY.csv')
X = data.drop('Drug',axis=1)
y = data['Drug']*2-1

X = pd.get_dummies(X)  
ada=My_AdaBoost()
ada.learn(X, y, base_alg=[GaussianNB(), DecisionTreeClassifier()], lr=1)
predictions=ada.predict(X)


