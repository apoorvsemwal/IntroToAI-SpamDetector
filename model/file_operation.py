from model import calculated_values as cv


def writeTrainedTuples(trained_tuples):
    with open("../files/model.txt", "w", encoding="utf-8") as trainData:
        for wordTuple in trained_tuples:
            count, word, tfSpam, spamProb, tfHam, hamProb = wordTuple
            trainData.write(
                str(count) + "  " + word + " " + str(tfSpam) + " " + str(round(spamProb, 6)) + " " + str(
                    tfHam) + " " + str(
                    round(hamProb, 6)) + "\n")


# one of the way to write the calculated values
def writeCalculatedValues():
    with open("../files/values.txt", "w", encoding="utf-8") as calculateInfo:
        for params in cv.valuesIndex:
            cv.valuesIndex[params] = getattr(cv, params)
            calculateInfo.write(params + " " + str(cv.valuesIndex.get(params)) + "\n")


# one of the way to read the calculated values, if previous trained model data is use
def readCalculatedValues():
    with open("../files/values.txt", "r", encoding="utf-8") as calculateInfo:
        for line in calculateInfo:
            key, value = line.split(" ", 2)
            setattr(cv, key, float(value))


def readTrainedModelFile():
    with open("../files/model.txt", "r") as trainData:
        for line in trainData:
            _, _, word, _, hamProb, _, spamProb = line.split(" ")
            cv.wordsWithProb[word] = (word, float(hamProb), float(spamProb))


def getPredictionFileObj():
    f = open("../files/result.txt", "w", encoding="utf-8")
    return f
