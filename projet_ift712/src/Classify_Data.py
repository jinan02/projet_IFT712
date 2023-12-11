from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, precision_score, recall_score, accuracy_score, f1_score


class Classify_Data:  
    def __init__(self): 
        """
        Constructor for the Classify_Data class
        """
        #Get all the classifiers that are available in sklearn library
        self.estimators = all_estimators(type_filter="classifier")

    def get_classifiers(self, six_classifiers): 
        """
        Creates an instance of each classifier with hyperparameters

        Args: 
            six_classifiers: List of dictionnaries that defines each classifier

        Returns:
            sk_classifiers: Same as output list but with a new key,in each dictionnary,
                            that has as value the classifier's instance
        """
        sk_classifiers = [] 
     
        for clf in six_classifiers:
            """
            loop on all the classifiers dictionnaries provided and use estimator_class to get the instance of the model
            """
            for estimator_name, estimator_class in self.estimators:  
                if estimator_name == clf['model']: 
                    estimator = estimator_class()
                    clf ['classifier'] = estimator
                    sk_classifiers.append(clf)
                    
        return sk_classifiers
        #return clf
    
    def train_classifier(self, X_train, y_train, classifiers, metric = "accuracy", fold_number=5): 
        """
        Train the six classifiers, using GridSearchCV to optimize the hyperparameters

        Args: 
            X_train: features used to train the model
            y_train: labels used to train the model
            classifiers: list of classifiers (same structure as the output of the get_classifiers fucntion)
            metric: metric on which will be based the gridsearchcv,(we choosed accuracy since we have balanced classes)
            fold_number: the number of folds of he cross validatonin the GridSearchCV

        Returns: 
            trained_classifiers: trained classifiers after the gridsearch
            best_scores: list of best accuracy of each model
            best_estimators: List of the instances of each model with the best hyperparameters
        """

        trained_calssifiers = []
        best_estimators = []
        best_scores = []
        
        for classifier_dict in classifiers:
            gridsearch = GridSearchCV(estimator = classifier_dict['classifier'],
                                    param_grid = classifier_dict['params'],
                                    scoring = metric,
                                    cv = fold_number)
            
            gridsearch.fit(X_train, y_train)

            trained_calssifiers.append(gridsearch)
            best_scores.append(gridsearch.best_score_)
            best_estimators.append(gridsearch.best_estimator_)

        return trained_calssifiers, best_scores, best_estimators

    
    def calculate_metrics(self, y_test, y_pred):
        """
        Calculates four metrics 

        Args: 
            y_test: reel data labels
            y_pred: predicions by the model

        Returns: 
            metrics: dictionnary containing the four metrics
        """
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred, average ="weighted")
        metrics['precision'] = precision_score(y_test, y_pred, average ="weighted")
        metrics['f1_score'] = f1_score(y_test, y_pred, average ="weighted")

        return metrics


    def predict_and_evaluate(self, X_test, y_test, trained_calssifiers, six_classifiers): 
        """
        Predict on test data

        Args:  
            X_test: features of test data 
            y_test: reel labels
            trained classifiers: List of the classifiers (output of the train_classifier function)
            six_classifiers: List of our six classifiers (In our case they're six, could be more)

        Returns: 
            evaluation_results: dictionnary that assciate each classifier to its calculated metrics
        """
        evaluation_results = {}

        for classifier , trained_classifier in zip(six_classifiers, trained_calssifiers): 
            classifier_name = classifier['model']
            y_pred = trained_classifier.predict(X_test)

            metrics =self.calculate_metrics(y_test, y_pred)
            evaluation_results[classifier_name] = metrics
        
        return evaluation_results
    
