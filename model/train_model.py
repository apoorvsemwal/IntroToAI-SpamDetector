from os import walk
from model import preprocessing
from model import calculated_values
from model import constants

spamWordsWithFreq = {}
hamWordsWithFreq = {}


def updateSpamAndHamVocab(filepath="", className=constants.HAM):
    with open(filepath, mode='r', encoding='iso-8859-1') as trainFile:
        for line in trainFile:
            # replace new line by empty space
            # line = str(line.encode('utf-8'), 'utf-8')
            line = preprocessing.cleaningSteps(line)
            if line != "":
                tokens = preprocessing.textToTokens(line.strip())
                updateWordsDictionary(tokens, className)


def updateWordsDictionary(tokens=None, className=""):
    for token in tokens:
        if token != '':
            if className == constants.SPAM:
                if token in spamWordsWithFreq:
                    count = spamWordsWithFreq.get(token) + 1
                    spamWordsWithFreq[token] = count
                else:
                    spamWordsWithFreq[token] = 1
                    calculated_values.wordsInSpam += 1
            else:
                if token in hamWordsWithFreq:
                    count = hamWordsWithFreq.get(token) + 1
                    hamWordsWithFreq[token] = count
                else:
                    hamWordsWithFreq[token] = 1
                    calculated_values.wordsInHam += 1


def readFilesFromDirectory(dirPath=""):
    noOfHamFiles = 0
    noOfSpamFiles = 0
    for (dirpath, _, filenames) in walk(dirPath):
        for name in filenames:
            if constants.HAM in name:
                noOfHamFiles += 1
                currentFileClass = constants.HAM
            else:
                noOfSpamFiles += 1
                currentFileClass = constants.SPAM
            updateSpamAndHamVocab(dirpath + name, currentFileClass)
            # break
    return [noOfHamFiles, noOfSpamFiles]


def calculateSpamHamEachWordClassProb(vocabulary):
    spam_file = open("spam_prb.txt", "w")
    ham_file = open("ham_prb.txt", "w")
    for word in vocabulary:
        spam_word_prob = calculateSpamWordProbability(spamWordsWithFreq.get(word))
        ham_word_prob = calculateHamWordProbability(hamWordsWithFreq.get(word))
        spam_file.write(constants.SPAM + " :" + word + " : " + str(round(spam_word_prob, 6)) + "\n")
        ham_file.write(constants.HAM + " :" + word + " : " + str(round(ham_word_prob, 6)) + "\n")
    spam_file.close()
    ham_file.close()


def calculateSpamWordProbability(word_count=0):
    word_count = 0 if word_count is None else word_count
    numerator = word_count + constants.smoothing
    denominator = calculated_values.wordsInSpam + calculated_values.vocabLen
    return numerator / denominator


def calculateHamWordProbability(word_count=0):
    word_count = 0 if word_count is None else word_count
    numerator = word_count + constants.smoothing
    denominator = calculated_values.wordsInHam + calculated_values.vocabLen
    return numerator / denominator


def calculateClassProbability():
    calculated_values.spamClassProbability = calculated_values.spamFilesCount / (
            calculated_values.spamFilesCount + calculated_values.hamFilesCount)
    calculated_values.hamClassProbability = calculated_values.hamFilesCount / (
            calculated_values.spamFilesCount + calculated_values.hamFilesCount)
    print(calculated_values.spamClassProbability, calculated_values.hamClassProbability)


def startTraining():
    trainingPath = "../train/"
    hamFilesCount, spamFilesCount = readFilesFromDirectory(dirPath=trainingPath)
    calculated_values.spamFilesCount = spamFilesCount
    calculated_values.hamFilesCount = hamFilesCount
    hamKeys = set(hamWordsWithFreq.keys())
    spamKeys = set(spamWordsWithFreq.keys())
    hamKeys = hamKeys.union(spamKeys)
    calculated_values.vocabLen = len(hamKeys)
    sorted(hamKeys)
    calculateSpamHamEachWordClassProb(hamKeys)


if __name__ == '__main__':
    startTraining()
