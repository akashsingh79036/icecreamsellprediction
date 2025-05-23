import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://akash:akash@cluster0.44hv4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['icecreamsell']
collection=db['icecreamsell_pred']

def load_model(): 
    with open("icecream_polynomial_model.pkl" , 'rb') as file :
        poly_model,poly_model1,poly_model2,poly_model3,poly_model4=pickle.load(file)
    return poly_model,poly_model1,poly_model2,poly_model3,poly_model4

def preprocessing_input_data(data):
    df1=pd.read_csv("F:\study material\data science\euroncourse\material\icecreamsellpoly\Ice-cream-selling-data-csv_6UnYJ.csv")
    x=df1[['Temperature (°C)']]
    y=df1[['Ice Cream Sales (units)']]
    return x,y

def predict_data(x_input):
    poly_model,poly_model1,poly_model2,poly_model3,poly_model4 = load_model()
    processed_data = pd.DataFrame({'Temperature (°C)':[x_input]})
    y_pred=poly_model.predict(processed_data)
    # Clamp to zero to avoid negative sales
    predicted_sales = max(0, y_pred[0])
    return predicted_sales

def main():
    st.title("icecream selling Prediction")
    st.write("enter temperature to know the sell of icecream")
    x=st.number_input("Temperature (°C)",min_value=0,max_value=30)
    if st.button('predict_sales'):
        y_pred = predict_data(x)
        user_data={
            'x': x,
            'y_pred': float(y_pred)
        }
        st.success(f"Predicted ice cream sales: {int(y_pred)} units")
        collection.insert_one(user_data)
        
if __name__=="__main__":
    main()