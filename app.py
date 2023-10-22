#App made with Streamlit

#Loading Modules Needed
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plost
import pickle

#URL of the API made with Flask
API = "http://127.0.0.1:5000"

MODEL_PATH = f'./model/house_price_model.pkl'
SCALER_PATH = f'./model/scaler.pkl'
IMG_SIDEBAR_PATH = "./assets/img.jpg"

#Function to load the Model and the Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

#Function to load the Iris Dataset
def get_clean_data():
  data = pd.read_csv("./dataset/house_price_dataset.csv")
  
  X = pd.get_dummies(data)

  return X

#Sidebar of the Streamlit App
def add_sidebar():
  st.sidebar.header("House Price Predictor `App 🏠`")
  image = np.array(Image.open(IMG_SIDEBAR_PATH))
  st.sidebar.image(image)
  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
  st.sidebar.write("This Artificial Intelligence App can Predicts the Price of a House Given their Corresponding Parameters.")

  st.sidebar.subheader('Select the Parameters of the House ✅:')
  
  data = get_clean_data()
  
  slider_labels = [
        ("Longitude", "longitude"),
        ("Latitude", "latitude"),
        ("Housing Median Age", "housing_median_age"),
        ("Total Rooms", "total_rooms"),
        ("Total Bedrooms", "total_bedrooms"),
        ("Population", "population"),
        ("Households", "households"),
        ("Median Income", "median_income"),
        ("<1 Hour Ocean", "ocean_proximity_<1H OCEAN"),
        ("Inland", "ocean_proximity_INLAND"),
        ("Island", "ocean_proximity_ISLAND"),
        ("Near Bay", "ocean_proximity_NEAR BAY"),
        ("Near Ocean", "ocean_proximity_NEAR OCEAN"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )

  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

  st.sidebar.markdown('''
  🧑🏻‍💻 Created by [Luis Jose Mendez](https://github.com/mendez-luisjose).
  ''')

  return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['median_house_value'], axis=1)
  X = pd.get_dummies(X)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

#Radar Chart Function
def get_radar_chart(input_data):
  input_data = get_scaled_values(input_data)
  
  categories = ['Longitude', 'Latitude', 'Housing Median Age', 'Total Rooms', 'Total Bedrooms', 'Population', 'Households', 'Median Income', '<1 Hour Ocean', 'Inland', 'Island', 'Near Bay', 'Near Ocean']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['longitude'], input_data['latitude'], input_data['housing_median_age'],
          input_data['total_rooms'], input_data['total_bedrooms'], input_data['population'],
          input_data['households'], input_data['median_income'], input_data['ocean_proximity_<1H OCEAN'],
          input_data['ocean_proximity_INLAND'], input_data['ocean_proximity_ISLAND'], input_data['ocean_proximity_NEAR BAY'],
          input_data['ocean_proximity_NEAR OCEAN']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig

#Receiving Prediction Results from the API
def add_predictions(input_data) :
    input_array = np.array(list(input_data.values())).reshape(1, -1).tolist()

    input_array_scaled = scaler.transform(input_array)
    result = model.predict(input_array_scaled)

    pred_result = round(result[0], 2)

    #Run first the api.py file and the paste the URL in the API Variable if you want to deploy the Model with Flask and uncomment the next lines

    #data = {'array': input_array}

    #resp = requests.post(API, json=data)
    
    #pred_result = resp.json()["Results"]["price_result"]
    
    pred_result = round(pred_result, 2)
    pred_result = f"{pred_result}$"

    st.markdown("### House Price Prediction 💸")
    st.write("<span class='diagnosis-label diagnosis price'>Machine Learning Model Result ✅:</span>",  unsafe_allow_html=True)
    
    _, col, _ = st.columns([0.2, 1, 0.2])
    
    with col:
        st.metric("House Price 🏚️:", f"{pred_result}", "Dollars ($)")

def main() :  
    st.set_page_config(
        page_title="House Price Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
  
    input_data = add_sidebar()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )    

    with st.container() :
        st.title("House Price Predictor 🏡")
        st.write("This App predicts using a XGBRegressor Machine Learning Model the Price in Dollars ($) of a House. You can also Update the measurements by hand using sliders in the sidebar.")
        st.markdown("<hr/>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('### Radar Chart of the Parameters 📊')
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
        st.markdown('### Probability Plot of the Regressor Model 📉')
        st.image("./assets/probability_plot_model.png")

        st.markdown("---", unsafe_allow_html=True)
        st.write("`This Artificial Intelligence can Assist for the Price of a House, but Should Not be used as a Substitute for a Final Diagnosis and Prediction.`")


    with col2:
        st.markdown('### Evaluation of the Model 📈')
        st.image("./assets/model_evaluation.png")
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        add_predictions(input_data)

        st.markdown("---", unsafe_allow_html=True)
        st.markdown('### Model Displot 📊')
        st.image("./assets/model_displot.png")
        

if __name__ == "__main__" :
    main()

    print("App Running!")
