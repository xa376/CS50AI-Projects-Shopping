import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Parallel with evidenceList, will contain the revenue csv values, 1 if Revenue is true, and 0 otherwise
    labelsList = []
    # Parallel with labelsList, will contain the csv values besides revenue
    evidenceList = []

    # Opens the csv file and populates the labels and evidence lists
    with open(filename, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile)

        # Skips header
        next(reader)

        # populates the labels and evidence lists
        for user in reader:

            # Will hold current users evidence
            userEvidenceList = []

            #- Administrative, an integer
            userEvidenceList.append(int(user[0]))

            #- Administrative_Duration, a floating point number
            userEvidenceList.append(float(user[1]))

            #- Informational, an integer
            userEvidenceList.append(int(user[2]))

            #- Informational_Duration, a floating point number
            userEvidenceList.append(float(user[3]))

            #- ProductRelated, an integer
            userEvidenceList.append(int(user[4]))

            #- ProductRelated_Duration, a floating point number
            userEvidenceList.append(float(user[5]))

            #- BounceRates, a floating point number
            userEvidenceList.append(float(user[6]))

            #- ExitRates, a floating point number
            userEvidenceList.append(float(user[7]))

            #- PageValues, a floating point number
            userEvidenceList.append(float(user[8]))

            #- SpecialDay, a floating point number
            userEvidenceList.append(float(user[9]))

            #- Month, an index from 0 (January) to 11 (December)
            userEvidenceList.append(
                0 if user[10] == "Jan" else
                1 if user[10] == "Feb" else
                2 if user[10] == "Mar" else
                3 if user[10] == "Apr" else
                4 if user[10] == "May" else
                5 if user[10] == "Jun" else
                6 if user[10] == "Jul" else
                7 if user[10] == "Aug" else
                8 if user[10] == "Sep" else
                9 if user[10] == "Oct" else
                10 if user[10] == "Nov" else
                11
            )

            #- OperatingSystems, an integer
            userEvidenceList.append(int(user[11]))

            #- Browser, an integer
            userEvidenceList.append(int(user[12]))

            #- Region, an integer
            userEvidenceList.append(int(user[13]))

            #- TrafficType, an integer
            userEvidenceList.append(int(user[14]))

            #- VisitorType, an integer 0 (not returning) or 1 (returning)
            userEvidenceList.append(1 if user[15] == "Returning_Visitor" else 0)

            #- Weekend, an integer 0 (if false) or 1 (if true)
            userEvidenceList.append(0 if user[16] == "FALSE" else 1)

            # Appends this users evidence list to total evidence list
            evidenceList.append(userEvidenceList)

            # Appends 1 if Revenue is true, and 0 otherwise.
            labelsList.append(0 if user[17] == "FALSE" else 1)

    return (evidenceList, labelsList)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    
    # Instantiates a model
    model = KNeighborsClassifier(n_neighbors = 1)

    # Trains the model
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # ===== Will hold counts in upcoming loops =====
    # Total visits with purchase
    purchaseTotal = 0
    # Total correctly predicted visits with purchases
    correctPurchaseTotal = 0
    # Total visits without a purchase
    noPurchaseTotal = 0
    # Total correctly predicted visits without a purchase
    correctNoPurchaseTotal = 0

    # For every visit, (label), counts purchases, lack of purchases, and correct predictions
    for i in range(len(labels)):
        # Purchase made
        if labels[i] == 1:
            purchaseTotal += 1
            # Correctly Predicted
            if predictions[i] == 1:
                correctPurchaseTotal += 1
        # No purchase made
        elif labels[i] == 0:
            noPurchaseTotal += 1
            # Correctly predicted
            if predictions[i] == 0:
                correctNoPurchaseTotal += 1

    # % Correctly predicted purchases
    sensitivity = float(correctPurchaseTotal) / purchaseTotal
    # % Correctly predicted none purchases
    specificity = float(correctNoPurchaseTotal) / noPurchaseTotal

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
