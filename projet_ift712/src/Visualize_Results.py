import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA


class Visualize_Results:  
    def __init__(self): 
        """
        Constructor for the Visualize_Results class
        """
        pass
    
    def plot_training_scores(self, best_scores, six_classifiers):
        """
        plot the accuracy scores after training for each model

        Args: 
            best_scores: list containing the best score for each classifier
            six_classifiers: list of our classifiers

        Returns: 
            None: generates a plot

        """
        classifier_names = [classifier['model'] for classifier in six_classifiers]
        df_scores = pd.DataFrame({"model": classifier_names, "score": best_scores})
        plt.figure(figsize=(10,6))
        bars = plt.bar(df_scores['model'], df_scores['score'], color = 'blue')
        plt.bar_label(bars, labels = [f'{val:.2%}' for val in df_scores['score']])

        plt.title('Best Scores for Classifiers')

        plt.xlabel('Classifier')
        plt.ylabel('Best Score')
        plt.xticks(rotation = 45, ha='right')
        plt.tight_layout

        plt.show()

    def plot_evaluation_results(self, evaluation_results): 
        """ 
        Plot all the metrics results for each classifier

        Args: 
            evaluation_results: dictionnary containing each classifier with its calculated metrics
        
        Returns: 
            None: generates a plot
        """
        df = pd.DataFrame(evaluation_results).transpose()

        for metric in df.columns: 
            plt.figure(figsize=(10, 6))
            bars = plt.bar(df.index, df[metric])
            plt.bar_label(bars, labels = [f'{val:.2%}' for val in df[metric]])
            plt.title(f'comparsion of {metric} for all classifiers')
            plt.xlabel('Classifier')
            plt.ylabel(metric)
            plt.xticks(rotation = 45, ha='right')
            plt.show()

    def plot_learning_curve(self, X_train, y_train, six_classifiers, scoring ='accuracy'): 
        """
        Plot the training and the test curve together based on the accuracy scores

        Args: 
            X_train: features used to train the model
            y_train: labels used to train the model
            six_classifiers: list of our classifiers
            scoring: metric that we'll use to plot our curves

        Returns: 
            None: generates a plot  
        """
        for classifier_dict in six_classifiers:  
            classifier = classifier_dict['classifier']
            train_sizes, train_scores, test_scores = learning_curve(
                classifier, X_train, y_train, scoring = scoring, n_jobs=-1, cv=5)

            train_scores_mean = train_scores.mean(axis=1)
            train_scores_std = train_scores.std(axis=1)
            test_scores_mean = test_scores.mean(axis=1)
            test_scores_std = test_scores.std(axis=1)

            plt.figure()
            plt.title(f"Learning Curve for {classifier_dict['model']}")
            plt.xlabel("Samples")
            plt.ylabel(scoring)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score")

            plt.legend(loc="best")
            plt.show()
            

    def plot_explained_variance(self, data):
        """
        Plot the cumulative expalined variance for different numbers of components

        Args: 
            data: the datarame we're working with
        
        Returns: 
            None: generates a plot
        """
        pca = PCA()
        pca.fit(data)
        explained_variance = pca.explained_variance_ratio_

        # Plot the explained variance
        plt.plot(np.cumsum(explained_variance))
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.title('Elbow Method for Optimal Number of Components')
        plt.show()