import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class Preprocess_Data: 
    def __init__(self, path): 
        """
        Constructor for the Preprocess_Data class
        """
        self.path = path
        self.data = None

    def load_data_from_csv(self):
        """
        Loads the data from a csv file

        Args: 
            path: the pathe of the data

        Returns:
            data: dataframe loaded
        """
        self.data = pd.read_csv(self.path)
        return self.data
    
    def normalize_data(self, df, columns): 
        """
        Normalize data using MinMaxScaler

        Args: 
            df: dataframe we are working with
            columns: dataframe containing only the columns to normalize 

        Returns:
            df_normalized: normalized dataFrame

        """
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns = columns)

        return df_normalized
        
    
    def label_encoding(self, labels): 
        """
        Encode the class labels

        Args: 
            labels: original class labels

        Returns:
            encoded_labels: class labels after encoding
        """
        le = LabelEncoder()
        le.fit(labels)
        encoded_labels =le.transform(labels)

        return encoded_labels

    def Split_data(self, df, target):
        """
        Split data to train and test 

        Args: 
            df: dataframe of all the features 
            target: name of the target

        Returns:
            X_train, X_test, y_train, y_test: train and test features and train and test labels
        """
        features = df.drop(target, axis =1)
        targets = df[target]
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.33, random_state=42, stratify = targets)

        return X_train, X_test, y_train, y_test

    def PCA(self, data , n_components):
        """
        Applicate PCA on our data 

        Args: 
            data: dataframe we are workin witj
            n_components: number of dimensions to keep after PCA

        Returns:
            final_df: dataframe after reduction of dimensions
        """
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data)
        principal_df = pd.DataFrame(data = principal_components, columns=[f"component_{i+1}" for i in range(n_components)])
        final_df = pd.concat([principal_df, data['species']], axis = 1)

        return final_df
    
