#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import streamlit as st
import fasttext 

data = pd.read_excel('dataset-augmented-joined-fixed-with-id.xlsx')

model_ft_clas = fasttext.load_model('modelo_fasttext_class.ftz')

st.title("Chatbot Dirección Nacional de Admisiones")

# Interacción con el usuario
usuario_pregunta = st.text_input("Hola! En qué puedo ayudarte hoy? \n")

if st.button("Enviar"):
    if usuario_pregunta:
        # Llama a tu modelo de chatbot previamente entrenado
        y = model_ft_clas.predict(usuario_pregunta)[0][0][9:]
        y = int(y)
    
        respuesta = data[data['respuesta_id'] == y]['respuesta'].values[0]
        st.write("Con gusto te ayudaré!")
        st.write(respuesta)
    else:
        st.warning("Por favor, ingresa una pregunta.")

