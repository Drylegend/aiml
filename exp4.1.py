import pandas as pd
import math

def entropy(column):
    values = column.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in values)

def info_gain(data, split_attr, target_attr):
    total_entropy = entropy(data[target_attr])
    values = data[split_attr].unique()
    weighted_entropy = sum(
        (len(subset) / len(data)) * entropy(subset[target_attr])
        for value in values
        for subset in [data[data[split_attr] == value]]
    )
    return total_entropy - weighted_entropy

def id3(data, features, target):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    if not features:
        return data[target].mode()[0]

    gains = {feat: info_gain(data, feat, target) for feat in features}
    best_feature = max(gains, key=gains.get)
    tree = {best_feature: {}}

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, [f for f in features if f != best_feature], target)
        tree[best_feature][value] = subtree

    return tree

def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = sample.get(attr)
    subtree = tree[attr].get(value)
    return classify(subtree, sample) if subtree else "Unknown"

# Sample usage
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
features = list(df.columns[:-1])
target = 'Play'

tree = id3(df, features, target)
print("Decision Tree:", tree)

new_sample = {'Outlook': 'Sunny', 'Temp': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'}
prediction = classify(tree, new_sample)
print("Prediction:", prediction)
