from src import calculated_values as cv
from src import constants


def writeModelData(modelData):
    with open(constants.RESULTS_PATH + "model.txt", "w", encoding="utf-8") as trainData:
        for wordTuple in modelData:
            count, word, tfHam, hamProb, tfSpam, spamProb = wordTuple
            word = word.replace(" ", "-")
            trainData.write(
                str(count) + "  " + word + " " + str(tfHam) + " " + str(round(hamProb, 6)) + " " + str(
                    tfSpam) + " " + str(round(spamProb, 6)) + "\n")


# one of the ways to write the calculated values
def writeCalculatedValues():
    with open(constants.RESULTS_PATH + "values.txt", "w", encoding="utf-8") as calculateInfo:
        for params in cv.valuesIndex:
            cv.valuesIndex[params] = getattr(cv, params)
            calculateInfo.write(params + " " + str(cv.valuesIndex.get(params)) + "\n")


# one of the ways to read the calculated values, if previously trained model data is used
def readCalculatedValues():
    with open(constants.RESULTS_PATH + "values.txt", "r", encoding="utf-8") as calculateInfo:
        for line in calculateInfo:
            key, value = line.split(" ", 2)
            setattr(cv, key, float(value))


def readTrainedModelFile():
    with open(constants.RESULTS_PATH + "model.txt", "r", encoding="utf-8") as trainData:
        for line in trainData:
            _, _, word, _, hamProb, _, spamProb = line.split(" ")
            word = word.replace("-", " ")
            cv.wordsWithProb[word] = (float(hamProb), float(spamProb))


def getPredictionFileObj():
    f = open(constants.RESULTS_PATH + "result.txt", "w", encoding="utf-8")
    return f
