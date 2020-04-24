import math
from os import walk

from src import calculated_values as cv
from src import constants
from src import file_operation as fileop
from src import pre_processing


def startPredicting(filepath=""):
    predictionFileObj = fileop.getPredictionFileObj()
    for (dirpath, _, filenames) in walk(filepath):
        for filename in filenames:
            processTestFile(dirpath + filename, predictionFileObj)
    predictionFileObj.close()
    print("Prediction results saved...")
    print("\n========= Prediction Results =========")
    print("Total ham predicted: ", cv.predicted_ham, "/", constants.TOTAL_TEST_HAM)
    print("Total spam predicted: ", cv.predicted_spam, "/", constants.TOTAL_TEST_SPAM)
    print("\n========= Confusion Matrix Params =========")
    print("True Positive Count: ", cv.TruePositive)
    print("False Positive Count: ", cv.FalsePositive)
    print("True Negative Count: ", cv.TrueNegative)
    print("False Negative Count: ", cv.FalseNegative)
    showEvaluationResults()


# formula reference
def showEvaluationResults():
    print("\n========= Evaluation Sheet =========")
    accuracy = (cv.TruePositive + cv.TrueNegative) / (
            cv.TruePositive + cv.FalsePositive + cv.FalseNegative + cv.TrueNegative)
    print("Accuracy : ", valueToPercentage(accuracy))
    precision = cv.TruePositive / (cv.TruePositive + cv.FalsePositive)
    print("Precision : ", valueToPercentage(precision))
    recall = cv.TruePositive / (cv.TruePositive + cv.FalseNegative)
    print("Recall : ", valueToPercentage(recall))
    f1_score = 2 * (recall * precision) / (recall + precision)
    print("F1-Score : ", valueToPercentage(f1_score))


def processTestFile(filePath="", predictionFileObj=None):
    tokens = pre_processing.getValidFileTokens(filePath)
    predictForSpamOrHam(filePath, set(tokens), predictionFileObj)


def predictForSpamOrHam(filepath, tokens, predictionFileObj):
    cv.predictionCounter += 1
    filename = filepath.split("/")[-1]
    fileRealClass = constants.HAM if constants.HAM in filename else constants.SPAM
    hamChances, spamChances = getPredictionValues(tokens)
    if hamChances > spamChances:
        cv.predicted_ham += 1
        predictionFileObj.write(
            str(cv.predictionCounter) + " " + filename + " " + constants.HAM + " " + str(hamChances) + " " + str(
                spamChances) + " " + fileRealClass + " " + (
                "right" if fileRealClass == constants.HAM else "wrong") + "\n")
        update_confusion_matrix_params(fileRealClass, constants.HAM)
    else:
        cv.predicted_spam += 1
        predictionFileObj.write(
            str(cv.predictionCounter) + " " + filename + " " + constants.SPAM + " " + str(hamChances) + " " + str(
                spamChances) + " " + fileRealClass + " " + (
                "right" if fileRealClass == constants.SPAM else "wrong") + "\n")
        update_confusion_matrix_params(fileRealClass, constants.SPAM)


def getPredictionValues(tokens):
    logHamClassProb = math.log(cv.hamClassProbability, 10)
    logSpamClassProb = math.log(cv.spamClassProbability, 10)
    wordsHamProb = 0.0
    wordsSpamProb = 0.0
    for token in tokens:
        if cv.wordsWithProb.get(token) is not None:
            hamProb, spamProb = cv.wordsWithProb.get(token)
            wordsHamProb += 0.0 if hamProb == 0.0 else math.log(hamProb, 10)
            wordsSpamProb += 0.0 if spamProb == 0.0 else math.log(spamProb, 10)
    return (logHamClassProb + wordsHamProb, logSpamClassProb + wordsSpamProb)


# Spam as positive and Ham as negative
def update_confusion_matrix_params(actual, predicted):
    if actual == constants.HAM and predicted == constants.HAM:
        cv.TrueNegative += 1
    if actual == constants.SPAM and predicted == constants.SPAM:
        cv.TruePositive += 1
    if actual == constants.HAM and predicted == constants.SPAM:
        cv.FalsePositive += 1
    if actual == constants.SPAM and predicted == constants.HAM:
        cv.FalseNegative += 1


def valueToPercentage(param):
    return round(param * 100, 3)
