from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("fraud_model.pkl","rb"))

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    bank = bank_map[request.form["bank"]]
    card = card_map[request.form["card"]]
    merchant = merchant_map[request.form["merchant"]]
    city = city_map[request.form["city"]]
    device = device_map[request.form["device"]]

    amount = float(request.form["amount"])
    international = int(request.form["international"])
    hour = int(request.form["hour"])

    data = [[bank,card,amount,merchant,city,device,international,hour]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "⚠ Fraudulent Transaction Detected"
    else:
        result = "✅ Normal Transaction"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)