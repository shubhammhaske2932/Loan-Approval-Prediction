from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from utils import get_gender_encode, get_dependents_encode, get_education_encode,get_married_encode,get_property_area_encode,get_self_emp_encode

model = pickle.load(open("loan_pred_model.pkl","rb"))

app = Flask(__name__)

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


    Self_Employed = get_self_emp_encode(Self_Employed)
    Property_Area = get_property_area_encode(Property_Area)
    Dependents = get_dependents_encode(Dependents)
    Education = get_education_encode(Education)
    Gender = get_gender_encode(Gender)
    Married = get_married_encode(Married)

    
    test_df = pd.DataFrame({"Gender" : [Gender], 'Married' : [Married], 'Dependents' : [Dependents], 'Education' : [Education],
       'Self_Employed' : [Self_Employed], 'ApplicantIncome' : [ApplicantIncome], 'CoapplicantIncome' : [CoapplicantIncome], 'LoanAmount':LoanAmount,
       'Loan_Amount_Term' : [Loan_Amount_Term], 'Credit_History' : [Credit_History], 'Property_Area' : [Property_Area]})

    model_output = model.predict(test_df)

    return jsonify ("Loan Prediction : ",int(model_output[0]))

 
if __name__ == '__main__':
    app.run(debug = True)