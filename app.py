import streamlit as st
from IPython.display import display, HTML
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle
from sklearn.preprocessing import PowerTransformer
import base64



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


# step 1 load the pickled model --> rb read binary

model = open("model_xgb.pickle","rb")
clf = pickle.load(model)
model.close()

# step2 get the user input from the front end

adults = st.number_input('No of Adults',0,4,step = 1)
children = st.slider('No of Children',0,10,1)
weekend_nights = st.slider("No of weekend nights",0,7,1)
week_nights = st.slider('No of week nights',0,17,1)
meal_plan = st.selectbox("Select a meal plan ", ('Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'))
car_parking = st.selectbox("Parking required or not ", (0,1))
room_type = st.selectbox("Type of room type reserved ", ('Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
                       'Room_Type 5', 'Room_Type 6', 'Room_Type 7'))
lead_time = st.number_input("Lead Time" , 0,443,1)
arrival_month = st.slider("Month of arrival " , 1,12,1)
arrival_date = st.slider("Date of arrival", 1,30,1)
segment_type = st.selectbox("Mode of Booking ", ('Online','Aviation','Offline','Corporate','Complementary'))
repeated_guest = st.selectbox("Repeat visit 0 --> NO , 1 --> Yes" , (0,1))
previous_cancellations = st.slider("No of previous cancellations", 0,13,1)
not_cancelled = st.slider("No of successful visits" , 0,58,1)
avg_price = st.slider("Price per room" , 0, 540, 10)
special_request = st.slider("Special requests if any" , 0,5,1)


# step3 : converting user input to model input

data = {'no_of_adults': adults,
        'no_of_children' : children, 
        'no_of_weekend_nights' : weekend_nights, 
        'no_of_week_nights': week_nights,
        'type_of_meal_plan' : meal_plan,
       'required_car_parking_space': car_parking,
        'room_type_reserved': room_type,
        'lead_time': lead_time,
       "market_segment_type": segment_type,
       "repeated_guest": repeated_guest,
       "no_of_previous_cancellations" : previous_cancellations,
       "no_of_previous_bookings_not_canceled" : not_cancelled,
       "avg_price_per_room": avg_price,
       "no_of_special_requests" : special_request,
       "arrival_month": arrival_month,
       "arrival_date": arrival_date}

input_data = pd.DataFrame([data])

prediction = clf.predict(input_data)
if st.button("Prediction"):
    if prediction == 0:
        st.subheader("Booking will be honoured")
    if prediction==1:
        st.subheader("Booking will be cancelled")
