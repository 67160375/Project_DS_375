import streamlit as st
import pandas as pd
import joblib

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Airline Satisfaction Predictor", page_icon="✈️", layout="wide")

# 2. โหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load('models/airline_satisfaction_model.pkl')

model = load_model()

# 3. ส่วนหัวของเว็บ (Header & Disclaimer)
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown("""
แอปพลิเคชันนี้ใช้ Machine Learning (Random Forest) เพื่อทำนายว่าผู้โดยสารจะ **'พึงพอใจ'** หรือ **'ไม่พึงพอใจ'** ต่อการบริการของสายการบิน
""")

# 4. ส่วนรับข้อมูลจากผู้ใช้ (Sidebar)
st.sidebar.header("📋 กรอกข้อมูลผู้โดยสาร")

# ใช้ form เพื่อป้องกันเว็บโหลดใหม่ทุกครั้งที่ขยับ input
with st.sidebar.form("input_form"):
    st.subheader("ข้อมูลส่วนตัว & เที่ยวบิน")
    gender = st.selectbox("เพศ", ["Male", "Female"])
    customer_type = st.selectbox("ประเภทลูกค้า", ["Loyal Customer", "disloyal Customer"])
    age = st.number_input("อายุ", min_value=1, max_value=100, value=30) # Input validation
    type_of_travel = st.selectbox("จุดประสงค์การเดินทาง", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("ชั้นโดยสาร", ["Business", "Eco", "Eco Plus"])
    flight_distance = st.number_input("ระยะทางบิน (ไมล์)", min_value=50, max_value=5000, value=500)
    
    st.subheader("เวลาที่ล่าช้า (นาที)")
    dep_delay = st.number_input("เครื่องออกช้า (Departure Delay)", min_value=0, max_value=1000, value=0)
    arr_delay = st.number_input("เครื่องถึงช้า (Arrival Delay)", min_value=0, max_value=1000, value=0)

    st.subheader("คะแนนความพึงพอใจ (0-5)")
    wifi = st.slider("บริการ Wi-Fi บนเครื่อง", 0, 5, 3)
    time_convenient = st.slider("เวลาบิน/เวลาถึง สะดวกเหมาะสม", 0, 5, 3)
    ease_booking = st.slider("ความง่ายในการจองตั๋วออนไลน์", 0, 5, 3)
    gate_location = st.slider("ตำแหน่งประตูขึ้นเครื่อง (เดินไกลไหม)", 0, 5, 3)
    food = st.slider("อาหารและเครื่องดื่ม", 0, 5, 3)
    online_boarding = st.slider("ความสะดวกในการทำบอร์ดดิ้งพาสออนไลน์", 0, 5, 3)
    seat = st.slider("ความนั่งสบายของเบาะที่นั่ง", 0, 5, 3)
    entertainment = st.slider("สื่อบันเทิงบนเครื่อง (หนัง, เพลง)", 0, 5, 3)
    onboard_service = st.slider("การบริการของพนักงานบนเครื่อง", 0, 5, 3)
    leg_room = st.slider("พื้นที่วางขา (กว้างพอมั้ย)", 0, 5, 3)
    baggage = st.slider("การจัดการสัมภาระ (กระเป๋ามาไว/ไม่พัง)", 0, 5, 3)
    checkin = st.slider("บริการที่เคาน์เตอร์เช็คอิน", 0, 5, 3)
    inflight_service = st.slider("บริการภาพรวมระหว่างเที่ยวบิน", 0, 5, 3)
    cleanliness = st.slider("ความสะอาดของห้องโดยสาร", 0, 5, 3)

    submitted = st.form_submit_button("🔮 ทำนายผล (Predict)")

# 5. ส่วนแสดงผลการทำนาย (Main Page)
if submitted:
    # จัดเตรียมข้อมูลให้ตรงกับตอนที่เทรนโมเดล
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [type_of_travel],
        'Class': [travel_class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [wifi],
        'Departure/Arrival time convenient': [time_convenient],
        'Ease of Online booking': [ease_booking],
        'Gate location': [gate_location],
        'Food and drink': [food],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat],
        'Inflight entertainment': [entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room],
        'Baggage handling': [baggage],
        'Checkin service': [checkin],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [dep_delay],
        'Arrival Delay in Minutes': [arr_delay]
    })

    # ทำนายผล (Predict) และความน่าจะเป็น (Probability)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("🎯 ผลการทำนาย (Prediction Result)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.success("✅ ผู้โดยสารมีแนวโน้มที่จะ **พึงพอใจ (Satisfied)**")
        else:
            st.error("❌ ผู้โดยสารมีแนวโน้มที่จะ **ไม่พึงพอใจ (Neutral or Dissatisfied)**")
            
    with col2:
        st.info(f"📊 **ความมั่นใจของโมเดล (Confidence):** \n\n"
                f"- โอกาสพึงพอใจ: {probability[1]*100:.1f}%\n"
                f"- โอกาสไม่พึงพอใจ: {probability[0]*100:.1f}%")

# 6. คำอธิบาย Features & Disclaimer (ตาม Rubric)
st.markdown("---")
with st.expander("ℹ️ คำอธิบายตัวแปร (Feature Explanations)"):
    st.markdown("""
    * **Customer Type:** ลูกค้าประจำ (Loyal) หรือ ลูกค้าขาจร (Disloyal)
    * **Class:** ชั้นโดยสาร เช่น Business, Eco, Eco Plus
    * **คะแนน 0-5:** เป็นการประเมินความพึงพอใจในแต่ละบริการ (0 = ไม่ได้ประเมิน, 1 = แย่ที่สุด, 5 = ดีเยี่ยม)
    * **Departure/Arrival Delay:** เวลาที่เครื่องบินล่าช้า (หน่วยเป็นนาที)
    """)

st.caption("⚠️ **Disclaimer:** แอปพลิเคชันนี้จัดทำขึ้นเพื่อเป็นโปรเจกต์รายวิชาการศึกษาเท่านั้น ผลลัพธ์ที่ได้มาจากโมเดล Machine Learning ไม่ควรนำไปใช้ในการตัดสินใจทางธุรกิจจริงโดยไม่มีการตรวจสอบเพิ่มเติม")