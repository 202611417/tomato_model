import streamlit as st
import pandas as pd
import joblib  


@st.cache_resource
def load_model():
    return joblib.load("tomato_model.pkl") 
    

rf_model = load_model()

# --- 웹 앱 UI 구성 ---
st.title("🌱 착과율 예측 프로그램")
st.write("온실 내부 환경 데이터를 입력하여 예상되는 착과율을 확인해보세요.")

st.subheader("1. 환경 데이터 입력")
# 화면을 3개의 열(Column)로 나누어 깔끔하게 배치합니다.
col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input("내부온도 입력 (°C)", value=25.0, step=0.1)
with col2:
    humidity = st.number_input("내부습도 입력 (%)", value=60.0, step=1.0)
with col3:
    soil_temp = st.number_input("지온 입력 (°C)", value=20.0, step=0.1)

# 예측 버튼 생성
if st.button("결과 예측하기"):
    # DataFrame으로 변환 (2차원 배열 형태)
    input_data = pd.DataFrame([[temp, humidity, soil_temp]], columns=['내부온도', '내부습도', '지온'])
    
    try:
        # 예측 수행
        predicted = rf_model.predict(input_data)
        
        # 결과 출력
        st.subheader("2. 예측 결과")
        st.success(f"예측 착과율 : **{predicted[0]:.1f}%**")
        
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다. 모델이 제대로 로드되었는지 확인해주세요.\n(에러 메시지: {e})")