import matplotlib.pyplot as plt
import numpy as np

class PlotService:

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