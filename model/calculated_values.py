hamFilesCount = 0
spamFilesCount = 0
spamClassProbability = 0.0
hamClassProbability = 0.0
vocabLen = 0
wordsInSpam = 0
wordsInHam = 0
spamWordsWithFreq = {}
hamWordsWithFreq = {}
wordsWithProb = {}

predictionCounter = 0
# the values index is use to read calculated information from file when model is already trained.
valuesIndex = {"hamFilesCount": hamFilesCount, "spamFilesCount": spamFilesCount,
               "spamClassProbability": spamClassProbability, "hamClassProbability": hamClassProbability,
               "vocabLen": vocabLen, "wordsInSpam": wordsInSpam, "wordsInHam": wordsInHam,
               }
