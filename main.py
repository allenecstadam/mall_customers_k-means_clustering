import pandas as pd
import os
os.environ["OMP_NUM_THREADS"] = "1"

from src.data_loader import load_data
from src.model import train_model, save_model



def main():
    
        df = load_data("data/mall_customers.csv")
        df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
        train_features = ["Age", "Annual_Income", "Spending_Score"]
        k = 5

        model, labels, inertia = train_model(df[train_features], k)
        save_model(model)

    



main()
