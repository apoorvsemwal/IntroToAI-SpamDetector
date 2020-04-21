import numpy as np
import matplotlib.pyplot as plt
from model import constants


def graph_information():
    plt.xlabel("Spam/ham")
    plt.ylabel("No. of files")
    plt.title("Prediction Results")


def predictionGraph(predicted_spam, predicted_ham, show_plot=False):
    labels = [constants.SPAM, constants.HAM]
    ind = np.arange(len(labels))
    bar_width = 0.2
    actual_test_spam_ham_files = [400, 400]
    predicted_spam_ham_files = [predicted_spam, predicted_ham]
    ax = plt.subplot(111)
    # create a bar of actual and predicted spam/ham
    actual_spam_ham_files = ax.bar(ind, actual_test_spam_ham_files, width=0.2, alpha=0.5, color='b', align='center')
    predicted_spam_ham_files = ax.bar(ind + bar_width, predicted_spam_ham_files, width=0.2, alpha=0.5, color='r',
                                      align='center')
    # show which bar color represents what
    ax.legend((actual_spam_ham_files[0], predicted_spam_ham_files[0]), ('Actual', 'Predicted'))
    # do not show x-coordinates
    ax.set_xticklabels([])
    graph_information()

    def auto_label(rects, class_labels):
        for rect, label in zip(rects, class_labels):
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, label + ' (%d) ' % int(h),
                    ha='center', va='bottom')

    auto_label(actual_spam_ham_files, labels)
    auto_label(predicted_spam_ham_files, labels)
    if show_plot:
        plt.show()
    plt.savefig(constants.FILES_PATH + "prediction_results.png")
    print("\n========= Graph Generated =========")
    print("Results image saved in file: " + constants.FILES_PATH + "prediction_results.png")
