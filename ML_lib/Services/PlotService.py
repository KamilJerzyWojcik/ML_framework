import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import display
from pandas.plotting import scatter_matrix

class PlotService:
    
    def __init__ (self):
        pass

    def plot_result_cross_validate(self, model_name, cross_validate_result, ylim=(0, 1.)):
        mean_train = np.round( np.mean(cross_validate_result['train_score']), 2 )
        mean_test = np.round( np.mean(cross_validate_result['test_score']), 2)
        
        plt.title('{0}: cross validation\nmean-train-acc:{1}\nmean-test-acc:{2}'.format(model_name, mean_train, mean_test))
        plt.plot( cross_validate_result['train_score'], 'r-o', label="train" )
        plt.plot( cross_validate_result['test_score'], 'g-o', label='test' )

        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('# of fold')
        
        plt.ylim(*ylim)
        plt.show()

    def plot_histograms(self, dataframe, bins=50):
        dataframe.hist(bins=bins, figsize=(20,15))
        plt.show()

    def plot_scatter_mattrix(self, dataframe, attributes):
        scatter_matrix(dataframe[attributes], figsize=(12, 8))

    def show_corr_matrix(self, dataframe, label):
        corr_matrix = dataframe.corr()
        display(np.abs(corr_matrix[label]).sort_values(ascending=False))

    def plot_2D_chart(self, dataframe, x_column, y_column):
        dataframe.plot(kind="scatter", x=x_column, y=y_column, alpha=0.1)
        plt.show()

    def plot_2D_chart_label(self, dataframe, column_x, column_y, label):
        dataframe.plot(kind="scatter", 
                          x=column_x, 
                          y=column_y, 
                          alpha=1, 
                          figsize=(10,7),
                          c=label, 
                          cmap=plt.get_cmap("viridis"), 
                          colorbar=True, 
                          sharex=False)
        plt.show()