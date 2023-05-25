import streamlit as st
from IPython.display import display, HTML
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle
import base64
from category_encoders import WOEEncoder
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer

with open("HotelImage.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
f"""
<style>
.stApp {{
    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover
}}
</style>
""",
unsafe_allow_html=True
)

st.title("Hotel Booking Cancellation Prediction")
st.markdown("Will this customer honour the booking? ")

# step 1

model = open('model_xgb.pickle', "rb")
clf = pickle.load(model)
model.close()

# step 2
adults = st.number_input('No. of Adults',0,4,step = 1)
children = st.number_input('No. of children', 0,10,1)
wnd = st.slider('No. of weekend nights', 0,6)
wn = st.slider('No. of weekend nights', 0,17)
tmp = st.selectbox('Type of Meal Plan', ('Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected')) 
car_parking = st.number_input("Parking required or not", 0,1 , 1)
room_type = st.selectbox("Type of room type reserved ", ('Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
                       'Room_Type 5', 'Room_Type 6', 'Room_Type 7'))
lead_time = st.number_input("Lead Time" , 0,443,1)
segment_type = st.selectbox("Mode of Booking ", ('Online','Aviation','Offline','Corporate','Complementary'))
repeated_guest = st.selectbox("Repeat visit 0 --> NO , 1 --> Yes" , (0,1))
previous_cancellations = st.slider("No of previous cancellations", 0,13,1)
not_cancelled = st.slider("No of successful visits" , 0,58,1)
avg_price = st.slider("Price per room" , 0, 540, 10)
special_request = st.slider("Special requests if any" , 0,5,1)
day = st.selectbox('Weekday or Weekend',('Weekend','Weekday'))

# step3 : converting user input to model input

data = {'no_of_adults': adults,
        'no_of_children' : children, 
        'no_of_weekend_nights' : wnd, 
        'no_of_week_nights': wn,
        'type_of_meal_plan' : tmp,
       'required_car_parking_space': car_parking,
        'room_type_reserved': room_type,
        'lead_time': lead_time,
       "market_segment_type": segment_type,
       "repeated_guest": repeated_guest,
       "no_of_previous_cancellations" : previous_cancellations,
       "no_of_previous_bookings_not_canceled" : not_cancelled,
       "avg_price_per_room": avg_price,
       "no_of_special_requests" : special_request,
       "Day":day}

input_data = pd.DataFrame([data])

prediction = clf.predict(input_data)

if st.button("Check your Status"):
    if prediction == 0:
        st.subheader("Booking will be honoured")
    if prediction==1:
        st.subheader("Booking will be cancelled")
