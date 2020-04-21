# COMP-6721: Intro to AI 
**Project 2-SpamDetection**

![Spam-Ham Classification](https://1.bp.blogspot.com/-R4PgHVRlJvg/WhrDWjWy1AI/AAAAAAAAdCs/CMrnBlGaf6kSzm4TOQPN7y2Pf6E-QpGUACLcBGAs/s400/ml.PNG)

[Project Report Link](https://docs.google.com/document/d/16nX3PFPTzmznw6adeIMsn5luUDFEHX495sF_vpjBqkA/edit?usp=sharing)

**Main file: train_model.py**

_**Python Files:**_

- **calculated_values.py:** stores the values computed by model during the training phase.
- **constants.py:** stores the constants.
- **train_model.py:** contains the function use to train the model (ex: calculate class probability, calculate word probability etc).
- **file_operation.py:** contains the function related to I/O operation.
- **naive_bayes.py:** contains the function use to predicted the spam/ham email based on the model training.
- **pre_processing.py:** contains the function use to perform cleaning steps on the raw email before feeding model to learn.
- **graph.py:** generate the prediction results graph (bar graph).


_**Files information:**_

- **model.txt:** A file which stores the tuples containing word,tf(ham),prob(ham),tf(spam),prob(spam).

- **values.txt:** A file which stores the in-memory value, so that during testing phase it can directly use the previous train values.

- **result.txt:** Stores the results as mentioned in the project description.