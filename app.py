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
st.markdown("---")

# 4. ส่วนรับข้อมูลจากผู้ใช้ (Main Content Area - Columns)
# ลบ st.sidebar.header ออก แล้วใช้ st.header แทน
st.header("📋 กรอกข้อมูลผู้โดยสาร")

# ใช้ form ครอบข้อมูลทั้งหมดตรงกลางหน้าเว็บ
with st.form("input_form"):
    # --- แบ่งเป็น 3 คอลัมน์หลัก ---
    col1, col2, col3 = st.columns(3)

    # คอลัมน์ที่ 1: ข้อมูลส่วนตัวและเที่ยวบินพื้นฐาน
    with col1:
        st.subheader("👤 ข้อมูลส่วนตัว & ✈️ เที่ยวบิน")
        gender = st.selectbox("เพศ", ["Male", "Female"])
        customer_type = st.selectbox("ประเภทลูกค้า", ["Loyal Customer", "disloyal Customer"])
        age = st.number_input("อายุ", min_value=1, max_value=100, value=30)
        type_of_travel = st.selectbox("จุดประสงค์การเดินทาง", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("ชั้นโดยสาร", ["Business", "Eco", "Eco Plus"])
        flight_distance = st.number_input("ระยะทางบิน (ไมล์)", min_value=50, max_value=5000, value=500)
        
        st.write("") # เพิ่มช่องว่างเล็กน้อย
        st.subheader("⏱️ เวลาที่ล่าช้า (นาที)")
        dep_delay = st.number_input("เครื่องออกช้า (Departure Delay)", min_value=0, max_value=1000, value=0)
        arr_delay = st.number_input("เครื่องถึงช้า (Arrival Delay)", min_value=0, max_value=1000, value=0)

    # คอลัมน์ที่ 2: คะแนนความพึงพอใจส่วนที่ 1 (บริการเชิงระบบ/ดิจิทัล)
    with col2:
        st.subheader("🛋️  คะแนนบริการ")
        st.write("(0 = ไม่ได้ประเมิน, 1 = แย่ที่สุด, 5 = ดีเยี่ยม)")
        wifi = st.slider("บริการ Wi-Fi บนเครื่อง", 0, 5, 3)
        online_booking = st.slider("ความง่ายในการจองตั๋วออนไลน์", 0, 5, 3)
        online_boarding = st.slider("ความสะดวกทำบอร์ดดิ้งพาสออนไลน์", 0, 5, 3)
        gate_location = st.slider("ตำแหน่งประตูขึ้นเครื่อง", 0, 5, 3)
        checkin_service = st.slider("บริการเคาน์เตอร์เช็คอิน", 0, 5, 3)
        baggage_handling = st.slider("การจัดการสัมภาระ", 0, 5, 3)

    # คอลัมน์ที่ 3: คะแนนความพึงพอใจส่วนที่ 2 (บริการบนเครื่องและความสบาย)
    with col3:
        st.subheader("🛋️ คะแนนบริการ ")
        st.write("(0 = ไม่ได้ประเมิน, 1 = แย่ที่สุด, 5 = ดีเยี่ยม)")
        time_convenient = st.slider("เวลาบิน/เวลาถึง สะดวกเหมาะสม", 0, 5, 3)
        food_drink = st.slider("อาหารและเครื่องดื่ม", 0, 5, 3)
        seat_comfort = st.slider("ความนั่งสบายของเบาะที่นั่ง", 0, 5, 3)
        leg_room = st.slider("พื้นที่วางขา (กว้างพอมั้ย)", 0, 5, 3)
        entertainment = st.slider("สื่อบันเทิงบนเครื่อง", 0, 5, 3)
        onboard_service = st.slider("การบริการพนักงานบนเครื่อง", 0, 5, 3)
        inflight_service = st.slider("บริการภาพรวมระหว่างเที่ยวบิน", 0, 5, 3)
        cleanliness = st.slider("ความสะอาด", 0, 5, 3)

    st.markdown("---")
    # ปุ่มกดส่งข้อมูล ปรับขนาดให้เต็มหน้าจอเพื่อให้เห็นเด่นชัด
    submitted = st.form_submit_button("🔮 ทำนายผล (Predict)", use_container_width=True)

# 5. ส่วนแสดงผลการทำนาย (Main Page below form)
if submitted:
    # จัดเตรียมข้อมูลให้ตรงกับตอนที่เทรนโมเดล (ตรวจสอบชื่อคอลัมน์ให้เหมือนเดิมเป๊ะ)
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [type_of_travel],
        'Class': [travel_class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [wifi],
        'Departure/Arrival time convenient': [time_convenient],
        'Ease of Online booking': [online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [dep_delay],
        'Arrival Delay in Minutes': [arr_delay]
    })

    # ทำนายผล
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("🎯 ผลการทำนาย (Prediction Result)")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.success("✅ ผู้โดยสารมีแนวโน้มที่จะ **พึงพอใจ (Satisfied)**")
        else:
            st.error("❌ ผู้โดยสารมีแนวโน้มที่จะ **ไม่พึงพอใจ (Neutral or Dissatisfied)**")
            
    with res_col2:
        st.info(f"📊 **ความมั่นใจของโมเดล (Confidence):** \n\n"
                f"- โอกาสพึงพอใจ: {probability[1]*100:.1f}%\n"
                f"- โอกาสไม่พึงพอใจ: {probability[0]*100:.1f}%")

# 6. ส่วนท้าย (คงเดิม)
st.markdown("---")
with st.expander("ℹ️ คำอธิบายตัวแปร (Feature Explanations)"):
    st.markdown("""
    * **Customer Type:** ลูกค้าประจำ (Loyal) หรือ ลูกค้าขาจร (Disloyal)
    * **Class:** ชั้นโดยสาร เช่น Business, Eco, Eco Plus
    * **คะแนน 0-5:** เป็นการประเมินความพึงพอใจในแต่ละบริการ (0 = ไม่ได้ประเมิน, 1 = แย่ที่สุด, 5 = ดีเยี่ยม)
    * **Departure/Arrival Delay:** เวลาที่เครื่องบินล่าช้า (หน่วยเป็นนาที)
    """)

st.caption("🚀 พัฒนาโดยใช้ Random Forest Classifier | Accuracy ~95% | Disclaimer: โปรเจกต์เพื่อการศึกษา")
