#Abdullah Dedeoglu 2001001046

import pandas as pd
import numpy as np

data = pd.read_excel('LungCancer.xlsx')

#print(class_labels)


def show_data(data):
    print("Showing Data Set:")
    print(data)

def fill_missing_all(data):
    # print("Filling in Missing Values ​​of All Data")
    filled_data = data.copy()
    for column in filled_data.columns:
        if filled_data[column].dtype == 'object' or filled_data[column].dtype == 'int64':
            mode_value = filled_data[column].mode()[0] 
            #print("The most frequently occurring value of " + str(column) + " is: " + str(mode_value))
            filled_data[column] = filled_data[column].replace('?', mode_value)  
    return filled_data

def fill_missing_column(data, column_name):
    #print("Filling in Missing Values of Column: " + column_name)
    filled_data = data.copy()
    if filled_data[column_name].dtype == 'object' or filled_data[column_name].dtype == 'int64':
        mode_value = filled_data[column_name].mode()[0] 
        print("The most frequently occurring value of " + column_name + " column is: " + str(mode_value))
        filled_data[column_name] = filled_data[column_name].replace('?', np.nan)  
        filled_data[column_name].fillna(mode_value, inplace=True)  
    return filled_data

def convert_to_numeric(filled_data):
    for column in filled_data.columns:
        if filled_data[column].dtype == 'object':
            filled_data[column] = pd.to_numeric(filled_data[column], errors='coerce')
    return filled_data

def normalize_all(filled_data):
    normalized_data = filled_data.copy()
    # print("Normalization of All Data Set:\n")
    for column in normalized_data.columns:
        if np.issubdtype(normalized_data[column].dtype, np.number):
            min_value = normalized_data[column].min()
            max_value = normalized_data[column].max()
            normalized_data[column] = (normalized_data[column] - min_value) / (max_value - min_value)
    return normalized_data

def normalize_column(filled_data, column_name):
    normalized_data = filled_data.copy()
    print("Normalization of Selected Column:\n")
    if np.issubdtype(normalized_data[column_name].dtype, np.number):
        min_value = normalized_data[column_name].min()
        max_value = normalized_data[column_name].max()
        #print("min value: "+str(min_value)+" max_value: "+str(max_value))
        normalized_data[column_name] = (normalized_data[column_name] - min_value) / (max_value - min_value)
    return normalized_data

class CustomNaiveBayes:
    def separate_by_class(self, X, y):
        separated = dict()
        for i in range(len(X)):
            features = X[i]
            class_label = y[i]
            if class_label not in separated:
                separated[class_label] = list()
            separated[class_label].append(features)
        return separated

    def calculate_mean_stddev(self, dataset):
        summaries = [(np.mean(column), np.std(column)) for column in zip(*dataset)]
        return summaries

    def fit(self, X, y):
        self.separated = self.separate_by_class(X, y)
        self.summaries = dict()
        for class_value, instances in self.separated.items():
            self.summaries[class_value] = self.calculate_mean_stddev(instances)

    def calculate_probability(self, x, mean, stdev):
        if np.isclose(stdev, 0):
            stdev = 1e-8  
            mean += stdev  
        
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, input_data):
        probabilities = dict()
        for class_value, class_summaries in self.summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                x = input_data[i]
                probabilities[class_value] *= self.calculate_probability(x, mean, stdev)
        return probabilities

    def predict(self, X_test):
        predictions = list()
        for i in range(len(X_test)):
            probabilities = self.calculate_class_probabilities(X_test[i])
            best_label, best_prob = None, -1
            for class_value, probability in probabilities.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = class_value
            predictions.append(best_label)
        return predictions

def classification(normalized_data):
    X = normalized_data.iloc[:, 1:]  
    y = normalized_data.iloc[:, 0]   

    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)

    X_train, X_test = X.values[:split_index], X.values[split_index:]
    y_train, y_test = y.values[:split_index], y.values[split_index:]

    print(f"Training dataset size: {len(X_train)}")
    print(f"Test dataset size: {len(X_test)}")

    nb_classifier = CustomNaiveBayes()  
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)
    return y_test, y_pred

def evaluation(y_test,y_pred):
    accuracy = sum(y_pred == y_test) / len(y_test)
    comparison = pd.DataFrame({'Authentic': y_test, 'Predicted': y_pred})
    return accuracy, comparison

def compare_results(filled_result, normalized_result):
    print("\nComparison of Results:")
    print("Filled Data Set Accuracy:", filled_result['accuracy'])
    print("Normalized Data Set Accuracy:", normalized_result['accuracy'])

    best_accuracy = max(filled_result['accuracy'], normalized_result['accuracy'])

    if best_accuracy == filled_result['accuracy']:
        print("\nBest Accuracy is achieved with the Filled Data Set.")
    else:
        print("\nBest Accuracy is achieved with the Normalized Data Set.")

def main():

    while True:
        print("\n--- Menü ---")
        print("1. Show Data Set")
        print("2. Fill Missing Values")
        print("3. Data Normalization")
        print("4. Classification")
        print("5. Evaluation")
        print("6. Exit")

        choice = input("Please select the action you want to take (1-6): ")

        if choice == '1':
            show_data(data)

        elif choice == '2':
            fill_choice = input("1. Fill all data set\n2. Fill a specific column\nChoose (1-2): ")
            if fill_choice == '1':
                filled_data=fill_missing_all(data)
                filled_data = convert_to_numeric(filled_data)
                print (filled_data)
            elif fill_choice == '2':
                column_name = input("Please enter the column name: ")
                filled_data=fill_missing_column(data, column_name)
                filled_data = convert_to_numeric(filled_data)
                print(filled_data[column_name])
            else:
                print("Invalid selection!")

        elif choice == '3':
            norm_choice = input("1. Normalize all columns\n2. Normalize a specific column\nChoose (1-2): ")
            if norm_choice == '1':
                normalized_data=normalize_all(filled_data)
                print(normalized_data)
            elif norm_choice == '2':
                column_name = input("Please enter the column name: ")
                normalized_data=normalize_column(filled_data, column_name)
                print(normalized_data[column_name])
            else:
                print("Invalid selection!")

        if choice == '4':
            if normalized_data is not None:
                # variances = normalized_data.var()
                # print("Feature Variances:")
                # print(variances)                
                nb_classifier = CustomNaiveBayes()
                # y_test_data, y_pred_data = classification(data)
                y_test_filled, y_pred_filled = classification(filled_data)
                y_test_normalized, y_pred_normalized = classification(normalized_data)
                
            else:
                print("The data set is not normalized. Please normalize first.")
        
        elif choice == '5':
            if normalized_data is not None:
                # accuracy_data, comparison_data = evaluation(y_test_data, y_pred_data)
                # print("Original Data Evaluation:")
                # print(f"Test seti doğruluğu: {accuracy_data}")
                # print("Karşılaştırma:")
                # print(comparison_data)

                accuracy_filled, comparison_filled = evaluation(y_test_filled, y_pred_filled)
                print("Filled Data Evaluation:")
                print(f"Test set accuracy: {accuracy_filled}")
                print("Comparison:")
                print(comparison_filled)

                accuracy_normalized, comparison_normalized = evaluation(y_test_normalized, y_pred_normalized)
                print("Normalized Data Evaluation:")
                print(f"Test set accuracy: {accuracy_normalized}")
                print("Comparison:")
                print(comparison_normalized)

                # evaluation_result_data = {'accuracy': accuracy_data, 'comparison': comparison_data}
                evaluation_result_filled = {'accuracy': accuracy_filled, 'comparison': comparison_filled}
                evaluation_result_normalized = {'accuracy': accuracy_normalized, 'comparison': comparison_normalized}

                compare_results(evaluation_result_filled, evaluation_result_normalized)
            else:
                print("The data set is not normalized. Please normalize first.")
            
        elif choice == '6':
            print("...Exit the program")
            break
        
        elif choice == '7':
            filled_data=fill_missing_all(data)
            filled_data = convert_to_numeric(filled_data)
            normalized_data=normalize_all(filled_data)

        else:
            print("You made an invalid choice. Please try again.")

if __name__ == "__main__":
    main()
