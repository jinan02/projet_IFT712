import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

class Preprocess_Data: 
    def __init__(self, path): 
        self.path = path
        self.data = None

    def load_data_from_csv(self):
        self.data = pd.read_csv(self.path)
        return self.data
    
    def normalize_data(self, df, columns): 
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns = columns)
        return df_normalized
        
    
    def label_encoding(self, labels): 
        le = LabelEncoder()
        le.fit(labels)
        encoded_labels =le.transform(labels)
        return encoded_labels

    def Split_data(self, df, target): 
        X = df.drop(target, axis =1)
        y = df[target]
        stratified = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=42)
        for train_index, test_index in stratified.split(X, y):  
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test
    