import re
import string


def cleaningSteps(fileContent=""):
    fileContent = fileContent.lower()
    #tokens = re.split('[^a-zA-Z]', fileContent)
    tokens = re.split('\W', fileContent)
    tokens = [word for word in tokens if word and len(word) > 2]
    return tokens


def getValidFileTokens(filePath):
    tokens = []
    with open(filePath, mode='r', encoding='iso-8859-1') as testFile:
        fileContent = testFile.read()
        fileContent = str(fileContent.encode('utf-8'), 'utf-8')
        if fileContent != "":
            tokens = cleaningSteps(fileContent)
    return tokens
