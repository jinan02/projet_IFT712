from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, precision_score, recall_score, accuracy_score, f1_score


class Classify_Data:  
    def __init__(self): 
        self.estimators = all_estimators(type_filter="classifier")

    def get_classifiers(self, six_classifiers): 
        sk_classifiers = [] 
        for clf in six_classifiers:
            for estimator_name, estimator_class in self.estimators:  
                if estimator_name == clf['model']: 
                    #create an instance of the classifier with hyperparameters
                    estimator = estimator_class()
                    #estimator = estimator_class
                    #sk_classifiers.append({clf['Classifier']: estimator})
                    clf ['classifier'] = estimator
                    sk_classifiers.append(clf)
                    
        return sk_classifiers
        #return clf
    
    def train_classifier(self, X_train, y_train, classifiers, metric = "accuracy", fold_number=5): 
        trained_calssifiers = []
        best_estimators = []
        best_scores = []
        for classifier_dict in classifiers:
            #classifier = list(classifier_dict.keys())[0]
            #hyperparmeters = classifier_dict[classifier]
            gridsearch = GridSearchCV(estimator = classifier_dict['classifier'],
                                    param_grid = classifier_dict['params'],
                                    scoring = metric,
                                    cv = fold_number)
            gridsearch.fit(X_train, y_train)
            trained_calssifiers.append(gridsearch)
            best_scores.append(gridsearch.best_score_)
            best_estimators.append(gridsearch.best_estimator_)
        return trained_calssifiers, best_scores, best_estimators

    '''def predict(self, X_test, trained_classifiers): 
        all_predictions = []
        for trained_classifier in trained_classifiers:
            predictions = trained_classifier.predict(X_test)
            all_predictions.append(predictions)
        return all_predictions'''
    
    def calculate_metrics(self, y_test, y_pred):
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred, average ="weighted")
        metrics['precision'] = precision_score(y_test, y_pred, average ="weighted")
        metrics['f1_score'] = f1_score(y_test, y_pred, average ="weighted")

        return metrics


    def predict_and_evaluate(self, X_test, y_test, trained_calssifiers, six_classifiers): 
        evaluation_results = {}

        for classifier , trained_classifier in zip(six_classifiers, trained_calssifiers): 
            classifier_name = classifier['model']
            y_pred = trained_classifier.predict(X_test)

            metrics =self.calculate_metrics(y_test, y_pred)
            evaluation_results[classifier_name] = metrics
        
        return evaluation_results
    
    def cross_validation(self, X, y, folds, classifier, metric): 
        scores = cross_val_score(classifier, X, y, cv = folds, scoring= metric)
        mean_accuracy = scores.mean()
        return mean_accuracy
    
