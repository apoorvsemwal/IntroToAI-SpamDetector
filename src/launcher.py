import os

import calculated_values as cv
import constants
import file_operation as fileop
import graph
import naive_bayes as nb
import train_model as tm


def main():
    # Model already trained
    fileop.readStopWords()
    if os.path.exists(constants.RESULTS_PATH + "model.txt"):
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


if __name__ == '__main__':
    main()
