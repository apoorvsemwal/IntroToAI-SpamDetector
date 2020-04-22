import os

from src import calculated_values as cv
from src import constants
from src import file_operation as fileop
from src import pre_processing


def createHamAndSpamVocab(filepath="", className=constants.HAM):
    tokens = pre_processing.getValidFileTokens(filepath)
    createWordFreqDictionary(tokens, className)


def createWordFreqDictionary(tokens=None, className=""):
    for token in tokens:
        if className == constants.SPAM:
            if token in cv.spamWordsWithFreq:
                count = cv.spamWordsWithFreq.get(token) + 1
                cv.spamWordsWithFreq[token] = count
            else:
                cv.spamWordsWithFreq[token] = 1
            cv.totalFreqInSpam += 1
        else:
            if token in cv.hamWordsWithFreq:
                count = cv.hamWordsWithFreq.get(token) + 1
                cv.hamWordsWithFreq[token] = count
            else:
                cv.hamWordsWithFreq[token] = 1
            cv.totalFreqInHam += 1


def readFilesFromDirectory(dirPath=""):
    noOfHamFiles = 0
    noOfSpamFiles = 0
    for (dirpath, _, filenames) in os.walk(dirPath):
        for name in filenames:
            if constants.HAM in name:
                noOfHamFiles += 1
                currentFileClass = constants.HAM
            else:
                noOfSpamFiles += 1
                currentFileClass = constants.SPAM
            createHamAndSpamVocab(dirpath + name, currentFileClass)
    return [noOfHamFiles, noOfSpamFiles]


def saveModelData(vocabulary):
    modelData = []
    count = 0
    for word in vocabulary:
        count += 1
        currentWordFreqInHam = 0 if cv.hamWordsWithFreq.get(word) is None else cv.hamWordsWithFreq.get(word)
        currentWordFreqInSpam = 0 if cv.spamWordsWithFreq.get(word) is None else cv.spamWordsWithFreq.get(word)
        hamWordProb, spamWordProb = calculateWordProbability(currentWordFreqInHam, currentWordFreqInSpam)
        modelData.append((count, word, currentWordFreqInHam, hamWordProb, currentWordFreqInSpam, spamWordProb))
    fileop.writeModelData(modelData)
    fileop.writeCalculatedValues()


def calculateWordProbability(wordFreqInHam=0, wordFreqInSpam=0):
    numeratorHam = wordFreqInHam + constants.SMOOTHING
    denominatorHam = cv.totalFreqInHam + (cv.vocabLen * constants.SMOOTHING)
    numeratorSpam = wordFreqInSpam + constants.SMOOTHING
    denominatorSpam = cv.totalFreqInSpam + (cv.vocabLen * constants.SMOOTHING)
    return [(numeratorHam / denominatorHam), (numeratorSpam / denominatorSpam)]


def calculateClassProbabilities():
    cv.spamClassProbability = cv.spamFilesCount / (
            cv.spamFilesCount + cv.hamFilesCount)
    cv.hamClassProbability = cv.hamFilesCount / (
            cv.spamFilesCount + cv.hamFilesCount)


def prepareAndSaveModelData():
    trainingPath = constants.TRAINING_FILES
    cv.hamFilesCount, cv.spamFilesCount = readFilesFromDirectory(dirPath=trainingPath)
    hamKeys = set(cv.hamWordsWithFreq.keys())
    spamKeys = set(cv.spamWordsWithFreq.keys())
    fullVocab = sorted(hamKeys.union(spamKeys))
    cv.vocabLen = len(fullVocab)
    calculateClassProbabilities()
    saveModelData(fullVocab)
