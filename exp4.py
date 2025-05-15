import pandas as pd
import math

class Node:
    def __init__(self, attribute=None, value=None, result=None):
        self.attribute = attribute
        self.value = value
        self.result = result
        self.children = {}

def ID3(examples, attributes, target_attribute):
    if len(set(examples[target_attribute])) == 1:
        return Node(result=examples[target_attribute].iloc[0])

    if not attributes:
        majority_class = examples[target_attribute].mode()[0]
        return Node(result=majority_class)

    best_attribute = choose_attribute(examples, attributes, target_attribute)
    tree = Node(attribute=best_attribute)

    for value in examples[best_attribute].unique():
        examples_i = examples[examples[best_attribute] == value].drop(columns=[best_attribute])
        subtree = ID3(examples_i, attributes - {best_attribute}, target_attribute)
        tree.children[value] = subtree

    return tree

def choose_attribute(examples, attributes, target_attribute):
    information_gains = {}
    entropy_S = calculate_entropy(examples[target_attribute])

    for attribute in attributes:
        entropy_attribute = 0
        for value in examples[attribute].unique():
            examples_i = examples[examples[attribute] == value]
            entropy_i = calculate_entropy(examples_i[target_attribute])
            entropy_attribute += (len(examples_i) / len(examples)) * entropy_i

        information_gains[attribute] = entropy_S - entropy_attribute

    return max(information_gains, key=information_gains.get)

def calculate_entropy(attribute_values):
    entropy = 0
    total_count = len(attribute_values)

    for value in attribute_values.unique():
        count = len(attribute_values[attribute_values == value])
        probability = count / total_count
        entropy -= probability * math.log2(probability)

    return entropy

def classify_example(example, tree):
    if tree.result is not None:
        return tree.result
    else:
        value = example[tree.attribute]
        if value in tree.children:
            return classify_example(example, tree.children[value])
        else:
            # If the value is not present in the training data, return the majority class
            return tree.children[list(tree.children.keys())[0]].result



#Sample input and output
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
attributes = set(df.columns) - {'PlayTennis'}
tree = ID3(df, attributes, 'PlayTennis')

print("Decision Tree:")
print(tree)

new_sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'}
classification = classify_example(new_sample, tree)
print("Classification for the new sample:", classification)
