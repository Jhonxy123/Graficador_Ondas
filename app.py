from flask import Flask, render_template, request, send_file
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sin interfaz gráfica
import matplotlib.pyplot as plt
import base64


app = Flask(__name__)

def generar_ruido(temperatura_K, bw, nivel_sistema_dbm=None, num_puntos=1024):
    k = 1.380649e-23  # constante de Boltzmann
    ruido_watts = k * temperatura_K * bw
    if nivel_sistema_dbm:
        ruido_watts += 10**((nivel_sistema_dbm - 30)/10)
    # Promedio con fluctuaciones gaussianas
    ruido = np.random.normal(loc=ruido_watts, scale=ruido_watts*0.1, size=num_puntos)
    return ruido

@app.route('/', methods=['GET', 'POST'])
def index():
    espectro_img = None
    mediciones = None

    if request.method == 'POST':
        # Leer parámetros generales
        T = float(request.form['temperatura'])
        Bw_ruido = float(request.form['bw_ruido'])
        nivel_ruido = request.form.get('nivel_ruido')
        nivel_ruido = float(nivel_ruido) if nivel_ruido else None

        # Leer datos de señales
        señales = []
        for i in range(1, 4):
            P = float(request.form[f'potencia{i}'])
            Bw = float(request.form[f'bw{i}'])
            Fc = float(request.form[f'fc{i}'])
            señales.append((P, Bw, Fc))

        # Generar malla de frecuencias
        num_puntos = 2048
        freqs = np.linspace(0, max([Fc+Bw/2 for _,Bw,Fc in señales])*1.2, num_puntos)
        espectro = np.zeros_like(freqs)
        for P, Bw, Fc in señales:
             # 1) Convertir P dBm a watts lineales
            P_lin = 10**((P - 30)/10)
            # 2) Calcular sigma para la gaussiana (FWHM = Bw)
            sigma = Bw / (3 * np.sqrt(2 * np.log(2)))
            # 3) Generar forma de campana
            gauss = P_lin * np.exp(-0.5 * ((freqs - Fc)/sigma)**2)
            # 4) Sumar al espectro total
            espectro += gauss

        # Agregar ruido
        ruido = generar_ruido(T, Bw_ruido, nivel_ruido, num_puntos)
        espectro += ruido

        # Cálculo de mediciones (pico y SNR)
        picos = [10*np.log10(np.max(espectro * mask)) + 30 for _,Bw,Fc in señales
                 for mask in [np.logical_and(freqs >= Fc - Bw/2, freqs <= Fc + Bw/2)]]
        ruido_fondo = np.mean(ruido)
        snr = [p - (10*np.log10(ruido_fondo) + 30) for p in picos]
        mediciones = list(zip(range(1,4), picos, snr))

        # Graficar
        fig, ax = plt.subplots()
        ax.plot(freqs, 10*np.log10(espectro)+30)
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Nivel (dBm)')
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        #espectro_img = buf.getvalue()
        espectro_img = base64.b64encode(buf.getvalue()).decode('utf-8')


    return render_template('index.html', img_data=espectro_img, mediciones=mediciones)

if __name__ == '__main__':
    app.run(debug=True)