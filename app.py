from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open("loan_pred_model.pkl","rb"))

app = Flask(__name__)

@app.route('/Hello')
def Hello():
    return 'Hello Good Morning'

@app.route('/Loan_Prediction')

def Loan_Prediction():
    data = request.get_json()
    Gender = data['User_Gender']
    Married = data['User_Married']
    Dependents = data['User_Dependents']
    Education = data['User_Education']
    Self_Employed = data['Uesr_Self_Employed']
    ApplicantIncome = data['User_ApplicantIncome']
    CoapplicantIncome = data['User_CoapplicantIncome']
    LoanAmount = data['Uesr_LoanAmount']
    Loan_Amount_Term = data['User_Loan_Amount_Term']
    Credit_History = data['Uesr_Credit_History']
    Property_Area = data['User_Property_Area']

    
    test_df = pd.DataFrame({"Gender" : [Gender], 'Married' : [Married], 'Dependents' : [Dependents], 'Education' : [Education],
       'Self_Employed' : [Self_Employed], 'ApplicantIncome' : [ApplicantIncome], 'CoapplicantIncome' : [CoapplicantIncome], 'LoanAmount':LoanAmount,
       'Loan_Amount_Term' : [Loan_Amount_Term], 'Credit_History' : [Credit_History], 'Property_Area' : [Property_Area]})

    model_output = model.predict(test_df)

    return jsonify ("Loan Prediction : ",int(model_output[0]))


@app.route("/test")
def test():
    data1 = request.get_json()
    name = data1['user_name']
    mobile_no = data1['user_mobile_no']
    
    return jsonify({"Name":name, "Contact":mobile_no})
 
if __name__ == '__main__':
    app.run(debug = True)