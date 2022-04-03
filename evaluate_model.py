from pipeline import load_pipeline, load_recordings, get_epochs
from sklearn.metrics import confusion_matrix
import numpy as np


def main():
    model_subject_name = 'Haggai2'
    data_subject_name = 'Haggai2'

    # model
    pipeline = load_pipeline(model_subject_name)

    # data
    raw, params = load_recordings(data_subject_name)
    epochs, labels = get_epochs(raw, params["trial_duration"], params["calibration_duration"])
    epochs = epochs.get_data()

    # evaluate
    predictions = pipeline.predict(epochs)

    # statistics
    print_statistics(labels, predictions)


def print_statistics(labels, predictions):
    conf_matrix = confusion_matrix(labels, predictions)
    print('confusion matrix (row=label, column=prediction):')
    print(conf_matrix)

    rates = calculate_true_and_false_rates(conf_matrix)
    print('true positive and false positive rates (row=label, column=true or false):')
    print(rates)


def calculate_true_and_false_rates(conf_matrix):
    rates = np.zeros((3, 2))

    # true positive
    for i in range(0, 3):
        row = conf_matrix[i]
        total_num_true = sum(row)
        num_hits = row[i]
        rates[i][0] = num_hits / total_num_true

    # false positive
    for i in range(0, 3):
        column = conf_matrix[:, i]
        num_hits = column[i]
        total_num_predictions = sum(column)
        num_false_pos = total_num_predictions - num_hits
        rates[i][1] = num_false_pos / total_num_predictions

    return rates


if __name__ == "__main__":
    main()
