hamFilesCount = 0
spamFilesCount = 0
spamClassProbability = 0.0
hamClassProbability = 0.0
vocabLen = 0
totalFreqInSpam = 0
totalFreqInHam = 0
predicted_spam = 0
predicted_ham = 0
spamWordsWithFreq = {}
hamWordsWithFreq = {}
wordsWithProb = {}

stopWords = []
predictionCounter = 0
# the values index is use to read calculated information from file when model is already trained.
valuesIndex = {"hamFilesCount": hamFilesCount, "spamFilesCount": spamFilesCount,
               "spamClassProbability": spamClassProbability, "hamClassProbability": hamClassProbability,
               "vocabLen": vocabLen, "totalFreqInSpam": totalFreqInSpam, "totalFreqInHam": totalFreqInHam,
               }

# for confusion matrix
TruePositive = 0
FalsePositive = 0
TrueNegative = 0
FalseNegative = 0
