# Project Instruction
In this project we mainly explore a kind of explainable mechine learning method called Decision Tree. It also has an advanced version named RandomForest. We will use it to choice alpha signals and explain which signal contributes most to predict returns. Finally we calulate an AI-alpha factor which expressed as an expoure of alpha factors, then veiw it performance.

### Rank Factor by Explain Model
I introduce applications from Decision Tree to RandomForest after that explain model and rank them bettwen correlation and return. The code in `rank_features.ipynb` file

### AI Factors
In `RandomForest_Explainable_model.ipynb` project, we pick up some important alpha factors and combine them by AI method.

1. Construct factor by zipline
2. Pick up factors as alpha by DecisionTree
3. Use pickup alpha as X and return as y to train RandomForest model, so tha the model can predict return by alpha factors some way as AI_factor.
4. Split overlapping samples into indepent part by some ways to train RandomForest model agin so that avoid overfit as best as I can. 
5. View the alpha performances.
