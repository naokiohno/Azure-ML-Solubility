
# Load libraries
import pandas as pd

# Load data
solTrainX = pd.read_csv('data/solTrainX.csv')
solTrainY = pd.read_csv('data/solTrainY.csv')
solTestX = pd.read_csv('data/solTestX.csv')
solTestY = pd.read_csv('data/solTestY.csv')

# Join datasets into a single training set
solTrain = pd.concat([solTrainX, solTrainY], axis=1)
solTest = pd.concat([solTestX, solTestY], axis=1)

solubility = pd.concat([solTrain, solTest])
pd.DataFrame.to_csv(solubility, 'data/solubility_full.csv', index=False)
