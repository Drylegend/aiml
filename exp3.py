import csv
def find_s_algorithm(examples):
    hypothesis = initialize_hypothesis(len(examples[0]) - 1)

    for example in examples:
        if example[-1] == '1':  # Assuming the last column is the class label
            for i, attribute_value in enumerate(example[:-1]):
                if hypothesis[i] != attribute_value:
                    hypothesis[i] = '?'

    return hypothesis
def initialize_hypothesis(num_attributes):
    return ['0'] * num_attributes
# Sample input and output
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', '1'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', '1'],
    ['Rainy', 'Cold', 'High', 'Weak', 'Warm', 'Change', '0'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', '1']
]
hypothesis_result = find_s_algorithm(training_data)
print("Hypothesis:", hypothesis_result)
