import pandas as pd
import matplotlib.pyplot as plt

class Visualize_Results:  
    def __init__(self): 
        pass
    
    def plot_training_scores(self, best_scores, six_classifiers):
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
    
    '''def calculate_metrics(self, y_test, y_pred):
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_test, y_pred, average ="weighted")
        metrics['recall'] = recall_score(y_test, y_pred, average ="weighted")
        metrics['precision'] = precision_score(y_test, y_pred, average ="weight")
        metrics['f1_score'] = f1_score(y_test, y_pred, average ="weight")

        return metrics'''
    
    '''def plot_evaluation_results(self, six_classifiers, evaluation_results): 
        classifier_names = [classifier['model'] for classifier in six_classifiers]
        df = pd.DataFrame(evaluation_results, index = classifier_names)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        fig.suptitle('Comparision of Evaluation Results')

        df['precision'].plot (kind = 'bar', ax=axes[0,0], title = 'Precision')
        axes[0, 0].set_ylabel('Precision Score')

        df['recall'].plot (kind = 'bar', ax=axes[0,1], title = 'Recall')
        axes[0, 1].set_ylabel('Recall Score')

        df['accuracy'].plot (kind = 'bar', ax=axes[0,0], title = 'Accuracy')
        axes[1, 0].set_ylabel('Accuracy Score')

        df['f1 score'].plot (kind = 'bar', ax=axes[0,0], title = 'F1 Score')
        axes[1, 1].set_ylabel('F1 Score')

        plt.tight_layout(rect=[0,0,1,0.96])
        plt.show() '''

    def plot_evaluation_results(self, evaluation_results): 
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

  