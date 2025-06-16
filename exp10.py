import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


data_dict = {
    'age':          [0, 0, 1, 4, 3, 2, 4, 0, 3, 1, 4, 4, 2, 2, 3, 0, 1, 3, 4],
    'Gender':       [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Family':       [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    'diet':         [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1],
    'Lifestyle':    [3, 3, 2, 3, 0, 1, 2, 3, 0, 0, 2, 3, 0, 1, 0, 0, 2, 0, 3],
    'cholestrol':   [0, 0, 1, 2, 2, 0, 0, 0, 2, 2, 0, 2, 0, 0, 1, 2, 1, 1, 2],
    'heartdisease': [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
}


heart_disease = pd.DataFrame(data_dict)


model = DiscreteBayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease'),
    ('diet', 'cholestrol')  
])


model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)


HeartDisease_infer = VariableElimination(model)

print('For age Enter { SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4 }')
print('For Gender Enter { Male:0, Female:1 }')
print('For Family History Enter { yes:1, No:0 }')
print('For diet Enter { High:0, Medium:1 }')
print('For lifeStyle Enter { Athlete:0, Active:1, Moderate:2, Sedentary:3 }')
print('For cholesterol Enter { High:0, BorderLine:1, Normal:2 }')


evidence = {
    'age': int(input('Enter age: ')),
    'Gender': int(input('Enter Gender: ')),
    'Family': int(input('Enter Family history: ')),
    'diet': int(input('Enter diet: ')),
    'Lifestyle': int(input('Enter Lifestyle: ')),
    'cholestrol': int(input('Enter cholestrol: '))
}


q = HeartDisease_infer.query(variables=['heartdisease'], evidence=evidence)
print(q)
