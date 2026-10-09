import json
from datetime import date, time, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Pronóstico de demanda eléctrica', page_icon='⚡', layout='centered')

CARPETA = Path(__file__).parent


@st.cache_resource
def cargar():
    modelo = joblib.load(CARPETA / 'modelo_demanda.joblib')
    with open(CARPETA / 'metadatos_modelo.json', encoding='utf-8') as f:
        meta = json.load(f)
    return modelo, meta


def crear_variables(fecha_hora, demanda_anterior, temperatura, humedad, festivo):
    """Mismas variables que en el entrenamiento (ver crear_variables del cuaderno)."""
    hora = fecha_hora.hour
    dia = fecha_hora.weekday()
    return pd.DataFrame([{
        'PreviousDemand': demanda_anterior,
        'Temperature': temperatura,
        'Humidity': humedad,
        'Hour_sin': np.sin(2 * np.pi * hora / 24),
        'Hour_cos': np.cos(2 * np.pi * hora / 24),
        'IsHoliday': int(festivo),
        'IsWeekend': int(dia >= 5),
        'DayOfWeek': dia,
        'Month': fecha_hora.month,
    }])


modelo, meta = cargar()

st.title('⚡ Pronóstico de demanda eléctrica horaria')
st.caption('Trabajo Final de Machine Learning · Sofía Castañeda Arias y James Jair Olarte López')

with st.form('entrada'):
    c1, c2 = st.columns(2)
    fecha = c1.date_input('Fecha a pronosticar', date(2023, 3, 1))
    hora = c2.time_input('Hora', time(18, 0), step=3600)
    demanda_anterior = st.number_input('Demanda de la hora anterior (MW)', 300.0, 2000.0, 900.0, step=10.0)
    c3, c4 = st.columns(2)
    temperatura = c3.slider('Temperatura pronosticada (°C)', -5.0, 45.0, 20.0, 0.5)
    humedad = c4.slider('Humedad relativa (%)', 0.0, 100.0, 60.0, 1.0)
    festivo = st.checkbox('Es día festivo')
    enviar = st.form_submit_button('Pronosticar', type='primary')

if enviar:
    fecha_hora = datetime.combine(fecha, hora)
    X = crear_variables(fecha_hora, demanda_anterior, temperatura, humedad, festivo)
    pred = float(modelo.predict(X)[0])
    mae = meta['metricas_prueba']['MAE (MW)']
    st.metric('Demanda estimada', f'{pred:,.1f} MW', f'{pred - demanda_anterior:+,.1f} MW vs hora anterior')
    st.write(f'Rango esperado (± MAE de prueba): **{pred - mae:,.0f} – {pred + mae:,.0f} MW**')

with st.expander('Sobre el modelo'):
    st.write(f"**Modelo:** {meta['modelo']}")
    st.write(f"**Entrenado con** {meta['registros_entrenamiento']} registros horarios "
             f"({meta['periodo'][0]} a {meta['periodo'][1]}).")
    st.write('**Métricas en el set de prueba:**')
    st.table(pd.DataFrame({'Modelo': meta['metricas_prueba'],
                           'Línea base (hora anterior)': meta['linea_base_hora_anterior']}))
