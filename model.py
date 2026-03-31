import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/indian_credit_card_fraud_dataset_5000.csv")

# Drop unnecessary columns
df = df.drop(["t_id","c_id"], axis=1)

# Convert time to hour
df["Transaction_Time"] = pd.to_datetime(df["Transaction_Time"])
df["Hour"] = df["Transaction_Time"].dt.hour
df = df.drop("Transaction_Time", axis=1)

# Manual mapping dictionaries
bank_map = {
"SBI":0,"HDFC":1,"ICICI":2,"Axis":3,
"Kotak":4,"PNB":5,"Bank of Baroda":6,"Canara Bank":7
}

card_map = {
"Visa":0,"MasterCard":1,"RuPay":2
}

merchant_map = {
"Grocery":0,"Electronics":1,"Restaurant":2,
"Travel":3,"Fuel":4,"Clothing":5,
"Online Shopping":6,"Pharmacy":7,
"Entertainment":8,"ATM Withdrawal":9,
"Food Delivery":10,"Ride Sharing":11
}

device_map = {
"Mobile":0,"POS":1,"Desktop":2,"ATM":3
}

city_map = {
"Kolkata":0,"Mumbai":1,"Delhi":2,"Bangalore":3,
"Chennai":4,"Hyderabad":5,"Pune":6,
"Ahmedabad":7,"Jaipur":8,"Lucknow":9
}

# Apply mappings
df["Bank"] = df["Bank"].map(bank_map)
df["Card_Type"] = df["Card_Type"].map(card_map)
df["Merchant_Category"] = df["Merchant_Category"].map(merchant_map)
df["Device_Type"] = df["Device_Type"].map(device_map)
df["Transaction_City"] = df["Transaction_City"].map(city_map)

# Features and target
X = df.drop("Is_Fraud", axis=1)
y = df["Is_Fraud"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

# Save model
pickle.dump(model,open("fraud_model.pkl","wb"))

print("Model trained successfully")