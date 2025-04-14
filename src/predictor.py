import pandas as pd



def predict_cluster(model, age, income, avg_score):
    input_df = pd.DataFrame([[age, income, avg_score]], columns=["Age", "Annual_Income", "Spending_Score"])
    prediction = model.predict(input_df)[0]
    return prediction