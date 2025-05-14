from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/grafica')
def grafica():
# Generar datos y gráfica
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]
    
    plt.plot(x, y)
    plt.title("Gráfica de ejemplo")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    
    # Convertir la gráfica a imagen en memoria
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convertir a base64 para incrustar en HTML
    imagen = base64.b64encode(buffer.getvalue()).decode()
    plt.close()  # Limpiar la figura
    
    return render_template('grafica.html', imagen=imagen)

@app.route('/grafica_interactiva')
def grafica_interactiva():
    # Parámetros de la distribución normal
    mu = 0      # Media
    sigma = 1    # Desviación estándar
    puntos = 1000  # Cantidad de puntos
    
    # Generar datos para la campana
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, puntos)
    y = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)
    
    # Crear gráfica
    fig = px.line(
        x = x,
        y = y,
        title = "Función de Campana de Gauss",
        labels = {"x": "Valor", "y": "Densidad de probabilidad"},
        color_discrete_sequence = ["#FF4B4B"]  # Color personalizado
    )
    
    # Añadir línea vertical en la media
    fig.add_vline(x=mu, line_dash="dash", line_color="grey")
    fig.update_layout(
        showlegend=True,
        hovermode="x unified"
    )
    
    # Convertir a JSON
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    print("JSON generado:", graph_json[:200])  # Imprime los primeros 200 caracteres
    return render_template('plotly.html', graph_json=graph_json)

@app.route('/grafica_chartjs')
def grafica_chartjs():
    # Parámetros
    mu = 0.0
    sigma = 1.0
    puntos = 200

    # Generar datos
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, puntos).round(3).tolist()
    y = ((1/(sigma * np.sqrt(2*np.pi))) *
         np.exp(-0.5 * ((np.array(x) - mu)/sigma)**2)
        ).round(5).tolist()

    # Pasamos x e y al template
    return render_template('chartjs.html', x_values=x, y_values=y)

if __name__ == '__main__':
    app.run(debug=True)