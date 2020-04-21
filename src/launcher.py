import os

from src import calculated_values as cv
from src import constants
from src import file_operation as fileop
from src import graph
from src import naive_bayes as nb
from src import train_model as tm

if __name__ == '__main__':
    # Model already trained
    if os.path.exists(constants.RESULTS_PATH + "src.txt"):
        fileop.readCalculatedValues()
        fileop.readTrainedModelFile()
    else:
        print("Preparing model data...")
        tm.prepareAndSaveModelData()
        print("Model data saved...")
        print("Start predicting test data...")
        fileop.readTrainedModelFile()
    testingFilePath = constants.TESTING_FILES
    nb.startPredicting(testingFilePath)
    graph.predictionGraph(cv.predicted_spam, cv.predicted_ham)
