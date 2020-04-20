from os import walk
import math
from model import calculated_values as cv
from model import pre_processing


def startPredicting(filepath=""):
    for (dirpath, _, filenames) in walk(filepath):
        for filename in filenames:
            processTestFile(dirpath + filename)


def processTestFile(filePath=""):
    with open(filePath, mode='r', encoding='iso-8859-1') as testFile:
        fileContent = ""
        for line in testFile:
            line = line.strip()
            if line != "":
                line = str(line.encode('utf-8'), 'utf-8')
                fileContent += pre_processing.cleaningSteps(line)
        tokens = pre_processing.textToTokens(fileContent)
        predictForSpamOrHam(filePath, set(tokens))


def predictForSpamOrHam(filename, tokens):
    spamChances = getHamPredictionValue(tokens)
    hamChances = getSpamPredictionValue(tokens)
    if hamChances > spamChances:
        print(filename, "ham")
    else:
        print(filename, "spam")


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
