import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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
        features = df.drop(target, axis =1)
        targets = df[target]
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.33, random_state=42, stratify = targets)
        return X_train, X_test, y_train, y_test
    