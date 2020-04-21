from os import walk
import math
from model import calculated_values as cv
from model import constants
from model import pre_processing
from model import file_operation as fp


def startPredicting(filepath=""):
    predictionFileObj = fp.getPredictionFileObj()
    for (dirpath, _, filenames) in walk(filepath):
        for filename in filenames:
            processTestFile(dirpath + filename, predictionFileObj)
    predictionFileObj.close()
    print("========= Prediction Results =========")
    print("Total spam predicted: ", cv.predicted_spam, "/", constants.TOTAL_TEST_SPAM)
    print("Total ham predicted: ", cv.predicted_ham, "/", constants.TOTAL_TEST_HAM)
    print("\n========= Confusion Matrix Params =========")
    print("True Positive", cv.TruePositive)
    print("False Positive", cv.FalsePositive)
    print("True Negative", cv.TrueNegative)
    print("False Negative", cv.FalseNegative)
    showPerformanceParams()


# formula reference
def showPerformanceParams():
    print("\n========= Performance Sheet =========")
    accuracy = (cv.TruePositive + cv.TrueNegative) / (
            cv.TruePositive + cv.FalsePositive + cv.FalseNegative + cv.TrueNegative)
    precision = cv.TruePositive / (cv.TruePositive + cv.FalsePositive)
    recall = cv.TruePositive / (cv.TruePositive + cv.FalseNegative)
    f1_score = 2 * (recall * precision) / (recall + precision)
    print("Accuracy : ", value_to_percentage(accuracy))
    print("Precision : ", value_to_percentage(precision))
    print("Recall : ", value_to_percentage(recall))
    print("F1-Score : ", value_to_percentage(f1_score))


def processTestFile(filePath="", predictionFileObj=None):
    with open(filePath, mode='r', encoding='iso-8859-1') as testFile:
        fileContent = ""
        for line in testFile:
            line = line.strip()
            if line != "":
                line = str(line.encode('utf-8'), 'utf-8')
                fileContent += pre_processing.cleaningSteps(line)
        tokens = pre_processing.textToTokens(fileContent)
        predictForSpamOrHam(filePath, set(tokens), predictionFileObj)


def predictForSpamOrHam(filepath, tokens, predictionFileObj):
    cv.predictionCounter += 1
    filename = filepath.split("/")[-1]
    fileRealClass = constants.HAM if constants.HAM in filename else constants.SPAM
    hamChances = getHamPredictionValue(tokens)
    spamChances = getSpamPredictionValue(tokens)
    if hamChances > spamChances:
        cv.predicted_ham += 1
        predictionFileObj.write(
            str(cv.predictionCounter) + " " + filename + " " + constants.HAM + " " + str(hamChances) + " " + str(
                spamChances) + " " + fileRealClass + " " + (
                "right" if fileRealClass == constants.HAM else "wrong") + "\n"
        )
        update_confusion_matrix_params(fileRealClass, constants.HAM)
    else:
        cv.predicted_spam += 1
        predictionFileObj.write(
            str(cv.predictionCounter) + " " + filename + " " + constants.SPAM + " " + str(hamChances) + " " + str(
                spamChances) + " " + fileRealClass + " " + ("right" if fileRealClass == constants.SPAM else "wrong") +
            "\n"
        )
        update_confusion_matrix_params(fileRealClass, constants.SPAM)


def getSpamPredictionValue(tokens):
    logClassProb = math.log(cv.spamClassProbability, 10)
    wordsTotalProb = 0.0
    for token in tokens:
        if cv.wordsWithProb.get(token) is not None:
            _, _, spamProb = cv.wordsWithProb.get(token)
            wordsTotalProb += math.log(spamProb, 10)
    return logClassProb + wordsTotalProb


def getHamPredictionValue(tokens):
    logClassProb = math.log(cv.hamClassProbability, 10)
    wordsTotalProb = 0.0
    for token in tokens:
        if cv.wordsWithProb.get(token) is not None:
            _, hamProb, _ = cv.wordsWithProb.get(token)
            wordsTotalProb += math.log(hamProb, 10)
    return logClassProb + wordsTotalProb


# ham as positive and spam as negative
def update_confusion_matrix_params(actual, predicted):
    if actual == constants.HAM and predicted == constants.HAM:
        cv.TruePositive += 1
    if actual == constants.SPAM and predicted == constants.SPAM:
        cv.TrueNegative += 1
    if actual == constants.HAM and predicted == constants.SPAM:
        cv.FalseNegative += 1
    if actual == constants.SPAM and predicted == constants.HAM:
        cv.FalsePositive += 1


def value_to_percentage(param):
    return round(param * 100, 3)
