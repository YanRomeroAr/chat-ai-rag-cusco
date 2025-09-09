import streamlit as st
import openai

st.title("🏛️ Asistente TUPA")

try:
    openai.api_key = st.secrets["openai_api_key"]
    
    if prompt := st.chat_input("Pregunta"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(response.choices[0].message.content)
except:
    st.error("Configura openai_api_key en secrets")
