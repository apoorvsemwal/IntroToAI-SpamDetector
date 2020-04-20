import os
from model import pre_processing
from model import calculated_values as cv
from model import constants
from model import file_operation as fp
from model import naive_bayes as nb


def updateSpamAndHamVocab(filepath="", className=constants.HAM):
    with open(filepath, mode='r', encoding='iso-8859-1') as trainFile:
        for line in trainFile:
            # replace new line by empty space
            line = str(line.encode('utf-8'), 'utf-8')
            line = pre_processing.cleaningSteps(line)
            if line != "":
                tokens = pre_processing.textToTokens(line)
                updateWordsDictionary(tokens, className)


def updateWordsDictionary(tokens=None, className=""):
    for token in tokens:
        if token != '' and len(token) >= 2:
            if className == constants.SPAM:
                if token in cv.spamWordsWithFreq:
                    count = cv.spamWordsWithFreq.get(token) + 1
                    cv.spamWordsWithFreq[token] = count
                else:
                    cv.spamWordsWithFreq[token] = 1
                    cv.wordsInSpam += 1
            else:
                if token in cv.hamWordsWithFreq:
                    count = cv.hamWordsWithFreq.get(token) + 1
                    cv.hamWordsWithFreq[token] = count
                else:
                    cv.hamWordsWithFreq[token] = 1
                    cv.wordsInHam += 1


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
            updateSpamAndHamVocab(dirpath + name, currentFileClass)
    return [noOfHamFiles, noOfSpamFiles]


def calculateSpamHamEachWordClassProb(vocabulary):
    trained_tuples = []
    count = 0
    for word in vocabulary:
        count += 1
        currentWordFreqInSpam = 0 if cv.spamWordsWithFreq.get(word) is None else cv.spamWordsWithFreq.get(word)
        currentWordFreqInHam = 0 if cv.hamWordsWithFreq.get(word) is None else cv.hamWordsWithFreq.get(word)
        spamWordProb = calculateSpamWordProbability(currentWordFreqInSpam)
        hamWordProb = calculateHamWordProbability(currentWordFreqInHam)
        trained_tuples.append((count, word, currentWordFreqInHam, hamWordProb, currentWordFreqInSpam, spamWordProb))
    trained_tuples = sorted(trained_tuples)
    fp.writeTrainedTuples(trained_tuples)


def calculateSpamWordProbability(word_count=0):
    word_count = 0 if word_count is None else word_count
    numerator = word_count + constants.SMOOTHING
    denominator = cv.wordsInSpam + cv.vocabLen
    return numerator / denominator


def calculateHamWordProbability(word_count=0):
    word_count = 0 if word_count is None else word_count
    numerator = word_count + constants.SMOOTHING
    denominator = cv.wordsInHam + cv.vocabLen
    return numerator / denominator


def calculateClassProbability():
    cv.spamClassProbability = cv.spamFilesCount / (
            cv.spamFilesCount + cv.hamFilesCount)
    cv.hamClassProbability = cv.hamFilesCount / (
            cv.spamFilesCount + cv.hamFilesCount)


def startTraining():
    trainingPath = "../train/"
    hamFilesCount, spamFilesCount = readFilesFromDirectory(dirPath=trainingPath)
    cv.spamFilesCount = spamFilesCount
    cv.hamFilesCount = hamFilesCount
    hamKeys = set(cv.hamWordsWithFreq.keys())
    spamKeys = set(cv.spamWordsWithFreq.keys())
    hamKeys = hamKeys.union(spamKeys)
    cv.vocabLen = len(hamKeys)
    sorted(hamKeys)
    calculateClassProbability()
    calculateSpamHamEachWordClassProb(hamKeys)


if __name__ == '__main__':
    # model already trained
    if os.path.exists("../files/model.txt"):
        fp.readCalculatedValues()
        fp.readTrainedModelFile()
    else:
        print("Training Model")
        startTraining()
        fp.writeCalculatedValues()
        fp.readTrainedModelFile()
    testingFilePath = "../test/"
    nb.startPredicting(testingFilePath)
