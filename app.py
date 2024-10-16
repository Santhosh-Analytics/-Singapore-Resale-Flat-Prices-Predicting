import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from scipy import stats
from scipy.stats import boxcox
import pickle
from datetime import date, timedelta
from streamlit_option_menu import option_menu
from scipy.special import inv_boxcox



# Set page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Singapore Flat resale price Predictor",
    page_icon=r'asset/icon.jpeg',
)


# Injecting HTML for custom styling

st.markdown(
    """
    <style>
    .custom-info-box {
        background-color: rgba(61, 157, 243, 0.2) !important;  /* Background color similar to st.info */
        padding: 10px;  /* Add some padding for spacing */
        border-left: 10px solid #1E88E5;  /* Add a colored border on the left */
        border-right: 0px solid #1E88E5;
        border-up: 10px solid #1E88E5;
        border-down: 10px solid #1E88E5;
        border-radius: 20px;  /* Rounded corners */
        font-family: Arial, sans-serif;  /* Font styling */
        font-size: 18px;  /* Font size adjustment */
        color: #ffffff;  /* Font color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

texts = """
<div class="custom-info-box">
    <strong>Information:</strong> Please update flat information to proceed.
</div>
"""

# Injecting CSS for custom styling
st.markdown("""
    <style>
    /* Tabs */
    div.stTabs [data-baseweb="tab-list"] button {
        font-size: 25px;
        color: #ffffff;
        background-color: #4CAF50;
        padding: 10px 20px;
        margin: 10px 2px;
        border-radius: 10px;
    }
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #009688;
        color: white;
    }
    div.stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3e8e41;
        color: white;
    }
    /* Button */
    .stButton>button {
        font-size: 22px;
        background-color: darkgreen;
        color: white;
        border: single;
        border-width: 5px;
        border-color: #3e8e41;
        padding: 10px 20px;
        text-align: center;
        text-decoration: text-overflow;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 22px;
    }
    .stButton>button:hover {
        background-color: forestgreen !important;
        color: white !important;
        border: signle !important;
        border-width: 5px !important;
        border-color: Darkgreen !important;
        font-size: 22px !important;
        text-decoration: text-overflow !important;
        transition: width 2s !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to perform Box-Cox transformation on a single value using a given lambda
def transform_single_value(value, lmbda):
    if value is None:
        return None  # Handle missing value
    transformed_value = boxcox([value], lmbda=lmbda)[0]
    return transformed_value

def reverse_boxcox_transform(predicted, lambda_val):
    return inv_boxcox(predicted, lambda_val)

# Load the saved lambda values
with open(r'pkls/boxcox_lambdas.pkl', 'rb') as f:
    lambda_dict = pickle.load(f)

    
with open(r'pkls/scale_reg.pkl', 'rb') as f:
    scale_reg = pickle.load(f)

with open(r'pkls/XGB_model.pkl', 'rb') as f:
    XGB_model = pickle.load(f)
    


with st.sidebar:
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Main Menu", ["About", 'Genie'], 
        icons=['house-door-fill', 'bar-chart-fill'], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#04AA6D"},
            "icon": {"color": "Coral", "font-size": "25px", "font-family": "Times New Roman"},
            "nav-link": {"font-family": "inherit", "font-size": "22px", "color": "#ffffff", "text-align": "left", "margin": "0px", "--hover-color": "#84706E"},
            "nav-link-selected": {"font-family": "inherit", "background-color": "saddlebrown", "color": "black", "font-size": "25px"},
        }
    )
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; font-size: 38px; color: #55ACEE; font-weight: 700; font-family: inherit;'>Your Singapore Resale Flat Price Genie</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid beige;'>", unsafe_allow_html=True)

if selected == "About":
    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Overview </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.
</p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Models </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        Regression model: XG Boost Regressor for predicting the continuous variable 'Flat price'.
    </p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Contributing </h3>", unsafe_allow_html=True)
    github_url = "https://github.com/Santhosh-Analytics/Singapore-Resale-Flat-Prices-Predicting"
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the <a href="{}">GitHub Repository</a>.
    </p>""".format(github_url), unsafe_allow_html=True)

if selected == "Genie":


    # Options for various dropdowns
    town_option = ['Tampines',  'Yishun',  'Jurong West',  'Bedok',  'Woodlands',  'Ang Mo Kio',  'Hougang',  'Bukit Batok',  'Choa Chu Kang',  'Bukit Merah',  'Pasir Ris',  'Sengkang',  'Toa Payoh',  'Queenstown',  'Geylang',  'Clementi',  'Bukit Panjang',  'Kallang/Whampoa',  'Jurong East',  'Serangoon',  'Bishan',  'Punggol',  'Sembawang',  'Marine Parade',  'Central Area',  'Bukit Timah',  'Lim Chu Kang']
    flat_type_option = ['4 Room',  '3 Room',  '5 Room',  'Executive',  '2 Room',  '1 Room',  'Multi Generation']
    flat_model_option =['Model A',  'Improved',  'New Generation',  'Simplified',  'Premium Apartment',  'Standard',  'Apartment',  'Maisonette',  'Model A2',  'Dbss',  'Model A-Maisonette',  'Adjoined Flat',  'Terrace',  'Multi Generation',  'Type S1',  'Type S2',  '2-Room',  'Improved-Maisonette',  'Premium Apartment Loft',  'Premium Maisonette',  '3Gen']
    lease_year_option ={1966,  1967,  1968,  1969,  1970,  1971,  1972,  1973,  1974,  1975,  1976,  1977,  1978,  1979,  1980,  1981,  1982,  1983,  1984,  1985,  1986,  1987,  1988,  1989,  1990,  1991,  1992,  1993,  1994,  1995,  1996,  1997,  1998,  1999,  2000,  2001,  2002,  2003,  2004,  2005,  2006,  2007,  2008,  2009,  2010,  2011,  2012,  2013,  2014,  2015,  2016,  2017,  2018,  2019,  2020}
    floor_option = [3,5]
    floor_no_option = None
    

    col1, col, col2 = st.columns([2,.5,2])

    with col1:
        town = st.selectbox('Select Town you are interested:', town_option, index=None, help="Select Town where you are looking for a property/flat", placeholder="Select Town where you are looking for a property/flat")
        flat_type = st.selectbox('Select type of Flat:', flat_type_option    , index=None, help="Select flat type you are interested.", placeholder="Select flat type you are interested.")
        flat_model = st.selectbox('Select Flat Model:', flat_model_option, index=None, help="Select flat model you like.", placeholder="Select flat model you like.")
        lease_year = st.selectbox('Select lease agreement year:', lease_year_option, index=None, help="The beginning of the lease term during which the tenant has the right to use and occupy the leased property.", placeholder="Starting point of a lease agreement.")
        

    with col2:

        floor_area = st.slider('Floor Area SQM:', min_value=20, max_value=500, value=65, help='Total Estimated space measured in square meters. Minimum value 20 sqm and maximum is 500 sqm.',)
        floor = st.selectbox('Select number of floors:', floor_option, index=None, help="Estimated number of floors.", placeholder="Estimated number of floors.")
        if floor ==3:
            floor_no_option = [number for number in range(3,52,3)]
        else:
            floor_no_option = [number for number in range(5,52,5)]
        floor_level = st.selectbox('Select top floor level: ', floor_no_option, index=None, help="Estimated range of floors.", placeholder="Estimated range of floors.")
        
        def check_conditions():
            return(
                town is not None and
                flat_type is not None  and
                flat_model is not None and
                lease_year is not None and
                floor_area > 0 and floor_area < 10000  and
                floor  is not None and
                floor_level is not None
            )
            
            
            
           
        st.write(' ')
        st.write(' ')
        button = st.button('Predict Flat Price!') if check_conditions() else st.markdown(texts,unsafe_allow_html=True)
    
    
    remaining_lease_year = lease_year + 99 - date.today().year if lease_year is not None else None
    floor_area_box = transform_single_value(floor_area, lambda_dict['floor_area_lambda'])     if floor_area is not None  else None
    town_mapping={'Lim Chu Kang': 1, 'Queenstown': 2, 'Ang Mo Kio': 3, 'Clementi': 4, 'Geylang': 5, 'Bedok': 6, 'Bukit Batok': 7, 'Yishun': 8, 'Toa Payoh': 9, 'Jurong East': 10, 'Central Area': 11, 'Jurong West': 12, 'Kallang/Whampoa': 13, 'Woodlands': 14, 'Hougang': 15, 'Serangoon': 16, 'Marine Parade': 17, 'Bukit Merah': 18, 'Bukit Panjang': 19, 'Tampines': 20, 'Choa Chu Kang': 21, 'Sembawang': 22, 'Pasir Ris': 23, 'Bishan': 24, 'Bukit Timah': 25, 'Sengkang': 26, 'Punggol': 27}
    year_mapping = {1990: 1, 1991: 2, 1992: 3, 1993: 4, 1994: 5, 1995: 6, 2002: 7, 2003: 8, 2004: 9, 2001: 10, 2005: 11, 2006: 12, 1999: 13, 2000: 14, 1998: 15, 1996: 16, 2007: 17, 1997: 18, 2008: 19, 2009: 20, 2010: 21, 2019: 22, 2015: 23, 2018: 24, 2011: 25, 2016: 26, 2017: 27, 2014: 28, 2020: 29, 2012: 30, 2013: 31, 2021: 32, 2022: 33, 2023: 34, 2024: 35}
    flat_type_mapping = {'1 Room': 1, '2 Room': 2, '3 Room': 3, '4 Room': 4, '5 Room': 5, 'Executive': 6, 'Multi Generation': 7}
    flat_model_mapping={'New Generation': 1, 'Standard': 2, 'Simplified': 3, 'Model A2': 4, '2-Room': 5, 'Model A': 6, 'Improved': 7, 'Improved-Maisonette': 8, 'Model A-Maisonette': 9, 'Premium Apartment': 10, 'Adjoined Flat': 11, 'Maisonette': 12, 'Apartment': 13, 'Terrace': 14, 'Multi Generation': 15, 'Premium Maisonette': 16, '3Gen': 17, 'Dbss': 18, 'Premium Apartment Loft': 19, 'Type S1': 20, 'Type S2': 21}
    #lease_year_mapping={1969: 1, 1971: 2, 1967: 3, 1968: 4, 1973: 5, 1970: 6, 1972: 7, 1974: 8, 1977: 9, 1980: 10, 1983: 11, 1975: 12, 1981: 13, 1976: 14, 1978: 15, 1979: 16, 1966: 17, 1982: 18, 1985: 19, 1984: 20, 1986: 21, 1987: 22, 1988: 23, 1990: 24, 1989: 25, 1991: 26, 1997: 27, 1998: 28, 1996: 29, 1999: 30, 1994: 31, 1993: 32, 2000: 33, 1995: 34, 1992: 35, 2001: 36, 2002: 37, 2003: 38, 2004: 39, 2012: 40, 2014: 41, 2015: 42, 2005: 43, 2007: 44, 2010: 45, 2013: 46, 2008: 47, 2016: 48, 2009: 49, 2017: 50, 2018: 51, 2019: 52, 2006: 53, 2020: 54, 2011: 55}
    # floor_mapping={1:0,2:.5,3: 1,4:1.5, 5: 2,6:2.5}
    floor_level_mapping={3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 5: 6, 18: 7, 10: 8, 21: 9, 24: 10, 20: 11, 27: 12, 25: 13, 35: 14, 40: 15, 30: 16, 33: 17, 36: 18, 39: 19, 42: 20, 45: 21, 48: 22, 51: 23}
    remaining_lease_year_mapping = {81: 1, 82: 2, 83: 3, 80: 4, 79: 5, 84: 6, 78: 7, 85: 8, 77: 9, 76: 10, 86: 11, 75: 12, 87: 13, 88: 14, 48: 15, 74: 16, 89: 17, 49: 18, 90: 19, 47: 20, 72: 21, 73: 22, 71: 23, 45: 24, 46: 25, 70: 26, 91: 27, 44: 28, 43: 29, 50: 30, 69: 31, 92: 32, 93: 33, 68: 34, 41: 35, 42: 36, 51: 37, 67: 38, 96: 39, 52: 40, 58: 41, 66: 42, 94: 43, 59: 44, 95: 45, 57: 46, 100: 47, 65: 48, 101: 49, 53: 50, 56: 51, 64: 52, 54: 53, 55: 54, 98: 55, 60: 56, 63: 57, 62: 58, 61: 59, 97: 60, 99: 61}
    town_median_list = {1: 9.323219299316406,2: 9.260793685913086,3: 9.80174732208252,4: 10.193489074707031,5: 9.80174732208252,6: 10.354616165161133,7: 10.716485977172852,8: 10.615007400512695,9: 9.446247100830078,10: 10.866004943847656,11: 9.446247100830078,12: 11.342947006225586,13: 9.859149932861328,14: 11.29664134979248,15: 11.203139305114746,16: 10.963958740234375,17: 9.916055679321289,18: 9.916055679321289,19: 11.29664134979248,20: 11.342947006225586,21: 11.525309562683105,22: 11.250040054321289,23: 12.172235488891602,24: 11.388961791992188,25: 11.342947006225586,26: 11.203139305114746,27: 10.91515064239502}
    

    



    # year1 = st.number_input('anasnd ')
    # remaining_lease_year = lease_year + 99 - year1 if lease_year is not None else None
    #remaining_lease_year = st.number_input('remaining_lease_year ')

    town=town_mapping[town] if town is not None else None
    year=year_mapping[date.today().year]
    flat_type=flat_type_mapping[flat_type] if flat_type is not None else None
    flat_model=flat_model_mapping[flat_model] if flat_model is not None else None
   # lease_year=lease_year_mapping[lease_year] if lease_year is not None else None
    # floor=floor_mapping[floor] if floor is not None else None
    floor_level=floor_level_mapping[floor_level] if floor_level is not None else None
    remaining_lease_year=remaining_lease_year_mapping[remaining_lease_year] if remaining_lease_year is not None else None
    
    
    location_specifics = floor_area_box * town if None not in (floor_area_box, town) else None
    #floor_area_year = floor_area_box / remaining_lease_yearj
    
    age = year - lease_year if None not in (year, lease_year) else None
    flat_area = flat_type * floor_area_box if None not in (flat_type, floor_area_box) else None
    model_area = flat_model * floor_area_box if None not in (flat_model, floor_area_box) else None
    town_mean_price = town_median_list[town] if None not in (town, town_median_list) else None
    # st.write(lease_year)
    floor_area_age = floor_area_box * (2024 - remaining_lease_year) if None not in (floor_area_box, lease_year) else None
    floor_weightage = (floor_level -  floor ) * floor_area_box if None not in (floor_level, floor) else None
    
    
    # def scale_conditions():
    #         return(
    #             town is not None and
    #             flat_type is not None  and
    #             flat_model is not None and
    #             lease_year is not None and
    #             floor_area > 0 and floor_area < 600  and
    #             floor  is not None and
    #             remaining_lease_year is not None and 
    #             year > 0 and
    #             floor_area_box > 0 and
    #             location_specifics > 0 and 
    #             floor_level is not None and
    #             age > 0 or age is not None and
    #             flat_area > 0 and
    #             model_area > 0 and
    #             town_mean_price > 0 and
    #             floor_area_age > 0 and
    #             floor_weightage > 0                
    #         )
    
    # year=year1
    
    data = np.array([[lease_year,floor,year,floor_area_box,town, flat_type, flat_model,remaining_lease_year,location_specifics,age,flat_area,model_area,town_mean_price,floor_area_age,floor_weightage]])
    st.write(data)
    # scaled_data = scale_reg.transform(data)
    # st.write(scaled_data)
    
    

    
    if button and (0 or None not in data):
        
        scaled_data = scale_reg.transform(data)   
        st.write(scaled_data)
        prediction = XGB_model.predict(scaled_data)
        st.write(prediction)
        lambda_val = lambda_dict['resale_price_lambda'] 
        transformed_predict=reverse_boxcox_transform(prediction, lambda_val) if data is not None else None
        rounded_prediction = round(transformed_predict[0],2)
        st.success(f"Based on the input, the Genie's predicted price is,  {rounded_prediction:,.2f}")
        st.info(f"On average, Genie's predictions are within approximately 10 to 20% of the actual market prices.")
        