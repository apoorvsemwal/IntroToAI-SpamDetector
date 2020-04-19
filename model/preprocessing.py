import re
import string


def cleaningSteps(rawText=""):
    rawText = rawText.lower()
    # rawText = re.split('\[\^a-zA-Z\]', rawText)
    rawText = re.sub(r"[\.:\/\/\n-@]", " ", rawText)
    rawText = rawText.replace("[", " ")
    rawText = rawText.replace("\\", " ")
    rawText = rawText.replace("]", " ")
    rawText = rawText.replace("'", " ")
    rawText = rawText.translate(rawText.maketrans('', '', string.punctuation))
    rawText = re.sub('\s+', ' ', rawText)
    # rawText = re.sub(r"\x08.", "", rawText)
    return rawText.strip()


def textToTokens(line=""):
    return line.split(" ")
