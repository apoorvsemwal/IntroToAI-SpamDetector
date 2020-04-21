# COMP-6721: Intro to AI 
**Project 2-SpamDetection**

![Spam-Ham Classification](https://1.bp.blogspot.com/-R4PgHVRlJvg/WhrDWjWy1AI/AAAAAAAAdCs/CMrnBlGaf6kSzm4TOQPN7y2Pf6E-QpGUACLcBGAs/s400/ml.PNG)

[Project Report Link](https://drive.google.com/file/d/1p9rSEnRzgfFPDgBrMYYb9S3sczY2pXq9/view)

**Main file: launcher.py**

_**Python Files:**_

- **calculated_values.py:** stores the values computed by model during the training phase.
- **constants.py:** stores the constants.
- **train_model.py:** contains the function use to train the model (Prepare Model Data) (ex: calculate class probability, calculate word probability etc).
- **file_operation.py:** contains the function related to I/O operation.
- **naive_bayes.py:** contains the function use to predict Spam/Ham label for an email, using the trained model.
- **pre_processing.py:** contains the function use to perform cleaning steps on the raw email before using it to train the model.
- **graph.py:** generate the prediction results graph (bar graph).


_**Files information:**_

- **model.txt:** File storing the tuples containing (word,tf(ham),prob(ham),tf(spam),prob(spam)). tf -> term frequency

- **result.txt:** File storing the final results based on the format mentioned in the project description.

- **values.txt:** File storing the intermediate count related values that can used directly to avoid recomputations everytime the code is executed.


_**Instructions to run the project:**_
* Download/Clone the Project Repo to your loacl machine - [IntroToAI-SpamDetector](https://github.com/apoorvsemwal/IntroToAI-SpamDetector.git)
* Note: Project can also be downloaded from google drive - [Google Drive Link](https://drive.google.com/drive/folders/1hFeO5xocprJfMTZcDSfcwEt-uOsAlrHS)

* Navigate to '\IntroToAI-SpamDetector\src' in your terminal

* Run CMD:
	python launcher.py
	
* Check results folder '\IntroToAI-SpamDetector\results'