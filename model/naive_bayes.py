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
    else:
        cv.predicted_spam += 1
        predictionFileObj.write(
            str(cv.predictionCounter) + " " + filename + " " + constants.SPAM + " " + str(hamChances) + " " + str(
                spamChances) + " " + fileRealClass + " " + ("right" if fileRealClass == constants.SPAM else "wrong") +
            "\n"
        )


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
