import re
from src import constants


def cleaningSteps(fileContent=""):
    fileContent = fileContent.lower()
    # tokens = re.split('[^a-zA-Z]', fileContent)
    tokens = re.split('\W', fileContent)
    return tokens


def getValidFileTokens(filePath):
    tokens = []
    with open(filePath, mode='r', encoding='iso-8859-1') as testFile:
        fileContent = testFile.read()
        fileContent = str(fileContent.encode('utf-8'), 'utf-8')
        if fileContent != "":
            tokens = cleaningSteps(fileContent)
            if constants.N_GRAM_VALUE == 1:
                tokens = getUnigramTokens(tokens)
            else:
                tokens = getBigramTokens(tokens)
    return tokens


def getUnigramTokens(tokens):
    return [word for word in tokens if word and len(word) > 2]


def getBigramTokens(tokens):
    bigramTokens = []
    i = 0
    while i < len(tokens) - 1:
        j = i + 1
        bigramTokens.append(tokens[i] + " " + tokens[j])
        i += 1
    return bigramTokens
