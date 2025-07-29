# -*- coding: utf-8 -*-
"""
Script de Benchmarking para evaluar y comparar el rendimiento y consumo de batería
entre sistemas operativos (ej. Windows vs. Linux) en el mismo hardware.

Este script simula una carga de trabajo mixta que incluye:
- Navegación web activa y realista con Selenium.
- Carga intensiva de CPU mediante cálculos matriciales con NumPy.
- Operaciones de I/O (lectura/escritura) en disco con compresión de archivos.

Los datos de rendimiento (CPU, RAM, Disco, Temperatura, Batería) se registran
en un archivo CSV para su posterior análisis y visualización.

Uso:
1. Instalar dependencias:
   pip install pandas matplotlib psutil selenium numpy

2. Descargar el ChromeDriver correspondiente a tu versión de Chrome:
   https://googlechromelabs.github.io/chrome-for-testing/

3. Ejecutar el script y dejarlo correr hasta que se complete o se detenga manualmente.
   python tu_script.py

**Para una comparativa justa, asegúrate de:**
- Empezar con la batería al 100% en ambos SO.
- Usar las mismas versiones de Python y librerías (usa un requirements.txt).
- Cerrar todas las demás aplicaciones.
- Mantener condiciones de red similares.
"""
import os
import time
import psutil
import threading
import platform
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# --- Configuración de la Simulación ---
DURATION_MINUTES = 120  # Duración total de la simulación en minutos
BATTERY_THRESHOLD_STOP = 20  # Detener si la batería baja de este %
MONITOR_INTERVAL_SECONDS = 30  # Intervalo para tomar datos de rendimiento

class SystemBenchmark:
    """
    Clase que encapsula la lógica de la simulación y el monitoreo.
    """
    def __init__(self, duration_minutes, battery_threshold):
        self.duration = duration_minutes * 60
        self.battery_threshold = battery_threshold
        self.stop_event = threading.Event()
        self.desktop_path = self._get_desktop_path()
        self.output_path = os.path.join(self.desktop_path, "benchmark_results")
        os.makedirs(self.output_path, exist_ok=True)

        # Identificador único para esta ejecución
        self.run_id = f"{platform.system().lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.csv_file = os.path.join(self.output_path, f"report_{self.run_id}.csv")
        self.log_file = os.path.join(self.output_path, f"log_{self.run_id}.txt")
        
        self.performance_data = []

    def _log(self, message):
        """Escribe un mensaje en el archivo de log y en la consola."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def _get_desktop_path(self):
        """Obtiene la ruta del escritorio de forma multiplataforma."""
        if platform.system() == "Windows":
            return os.path.join(os.environ["USERPROFILE"], "Desktop")
        else:
            # Para Linux y macOS, 'Desktop' suele estar en inglés.
            # Se podría añadir más lógica si se necesita soportar otros idiomas.
            return os.path.join(os.path.expanduser("~"), "Desktop")

    # --- Tareas de Simulación ---

    def _task_web_browsing(self):
        """Simula navegación web realista usando Selenium."""
        self._log("Iniciando tarea: Navegación Web")
        sites = [
            "https://www.wikipedia.org/wiki/Technology",
            "https://github.com/trending",
            "https://stackoverflow.com/questions",
            "https://www.theverge.com",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Un clásico para video
        ]
        
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--window-size=1920,1080")
            
            # Asegúrate de que chromedriver esté en el PATH o especifica la ruta
            # service = Service(executable_path='/path/to/your/chromedriver')
            # driver = webdriver.Chrome(service=service, options=options)
            driver = webdriver.Chrome(options=options)

            while not self.stop_event.is_set():
                for site in sites:
                    if self.stop_event.is_set(): break
                    self._log(f"Navegando a: {site}")
                    driver.get(site)
                    # Simular scroll para cargar contenido dinámico
                    for _ in range(3):
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)
                    time.sleep(10) # Tiempo de "lectura" en la página
            driver.quit()
        except Exception as e:
            self._log(f"[ERROR] en Navegación Web: {e}")

    def _task_cpu_intensive(self):
        """Simula carga de CPU con cálculos matriciales (NumPy)."""
        self._log("Iniciando tarea: Carga de CPU Intensiva")
        while not self.stop_event.is_set():
            try:
                # Multiplicación de matrices, una operación computacionalmente costosa
                matrix_a = np.random.rand(1000, 1000)
                matrix_b = np.random.rand(1000, 1000)
                result = np.dot(matrix_a, matrix_b)
                self._log("Cálculo matricial completado.")
                time.sleep(5) # Pequeña pausa
            except Exception as e:
                self._log(f"[ERROR] en Carga de CPU: {e}")

    def _task_disk_io(self):
        """Simula operaciones de I/O en disco (crear, escribir, comprimir, borrar)."""
        self._log("Iniciando tarea: Operaciones de Disco I/O")
        temp_dir = os.path.join(self.output_path, "temp_io")
        os.makedirs(temp_dir, exist_ok=True)
        
        while not self.stop_event.is_set():
            try:
                # 1. Crear un archivo grande con datos aleatorios
                file_path = os.path.join(temp_dir, f"data_{int(time.time())}.bin")
                self._log(f"Creando archivo: {file_path}")
                with open(file_path, "wb") as f:
                    f.write(os.urandom(50 * 1024 * 1024)) # 50 MB

                # 2. Comprimir el archivo
                zip_path = file_path.replace(".bin", ".zip")
                self._log(f"Comprimiendo a: {zip_path}")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(file_path, os.path.basename(file_path))

                # 3. Limpiar
                os.remove(file_path)
                os.remove(zip_path)
                self._log("Limpieza de archivos temporales completada.")
                time.sleep(15)
            except Exception as e:
                self._log(f"[ERROR] en Operaciones de Disco: {e}")
                time.sleep(15)


    # --- Monitoreo y Ejecución ---
    
    def _monitor_performance(self):
        """Toma una instantánea del estado del sistema."""
        timestamp = datetime.now()
        
        # Batería
        battery = psutil.sensors_battery()
        if battery is None:
            self._log("[ADVERTENCIA] No se detectó batería.")
            # Si no hay batería, no podemos cumplir la condición de parada.
            # El script se detendrá por tiempo.
            batt_percent, batt_plugged = -1, "N/A"
        else:
            batt_percent, batt_plugged = battery.percent, battery.power_plugged

        # CPU, RAM, Disco
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        # Temperatura (manejo robusto de errores)
        try:
            temps = psutil.sensors_temperatures()
            if platform.system() == "Linux" and 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
            elif platform.system() == "Windows" and 'wmi' in temps: # Intento para Windows
                 cpu_temp = temps['wmi'][0].current
            else: # Fallback genérico
                cpu_temp = next(iter(temps.values()))[0].current if temps else "N/A"
        except (IndexError, KeyError, Exception):
            cpu_temp = "N/A"

        data_row = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "battery_percent": batt_percent,
            "power_plugged": batt_plugged,
            "cpu_usage_percent": cpu_percent,
            "ram_usage_percent": ram_percent,
            "disk_usage_percent": disk_percent,
            "cpu_temp_celsius": cpu_temp
        }
        
        self.performance_data.append(data_row)
        self._log(f"MONITOR: Bat: {batt_percent}% | CPU: {cpu_percent}% | RAM: {ram_percent}% | Temp: {cpu_temp}°C")

        # Condición de parada por batería
        if not batt_plugged and batt_percent <= self.battery_threshold:
            self._log(f"Batería por debajo del umbral ({self.battery_threshold}%). Deteniendo simulación.")
            self.stop_event.set()

    def _generate_report(self):
        """Guarda los datos en CSV y genera una gráfica."""
        if not self.performance_data:
            self._log("No se recolectaron datos, no se puede generar el reporte.")
            return

        # Guardar CSV
        df = pd.DataFrame(self.performance_data)
        df.to_csv(self.csv_file, index=False, encoding='utf-8')
        self._log(f"Reporte CSV guardado en: {self.csv_file}")

        # Generar Gráfica
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # Eje Y izquierdo para porcentajes
        ax1.set_xlabel('Tiempo de Simulación')
        ax1.set_ylabel('Uso (%)', color='tab:blue')
        ax1.plot(df['timestamp'], df['battery_percent'], label='Batería (%)', color='green', linewidth=2.5)
        ax1.plot(df['timestamp'], df['cpu_usage_percent'], label='CPU (%)', color='red', linestyle='--')
        ax1.plot(df['timestamp'], df['ram_usage_percent'], label='RAM (%)', color='blue', linestyle=':')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.set_ylim(0, 105)

        # Eje Y derecho para temperatura
        # Solo graficar si hay datos numéricos de temperatura
        df_temp = df[pd.to_numeric(df['cpu_temp_celsius'], errors='coerce').notna()]
        if not df_temp.empty:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Temperatura (°C)', color='tab:orange')
            ax2.plot(df_temp['timestamp'], pd.to_numeric(df_temp['cpu_temp_celsius']), label='Temperatura CPU (°C)', color='orange', linestyle='-.')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.set_ylim(bottom=20)

        fig.suptitle(f'Análisis de Rendimiento y Batería - {platform.system()}', fontsize=16)
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        graph_file = os.path.join(self.output_path, f"grafica_{self.run_id}.png")
        plt.savefig(graph_file)
        self._log(f"Gráfica generada en: {graph_file}")


    def run(self):
        """Inicia y gestiona la simulación."""
        self._log("="*50)
        self._log(f"INICIANDO BENCHMARK EN {platform.system()} {platform.release()}")
        self._log(f"Duración máxima: {self.duration/60} minutos.")
        self._log(f"Umbral de parada por batería: {self.battery_threshold}%")
        self._log("="*50)

        # Iniciar hilos de tareas
        tasks = [
            self._task_web_browsing,
            self._task_cpu_intensive,
            self._task_disk_io
        ]
        threads = [threading.Thread(target=t, daemon=True) for t in tasks]
        for t in threads:
            t.start()
        
        start_time = time.time()
        while not self.stop_event.is_set():
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.duration:
                self._log("Tiempo de simulación completado.")
                self.stop_event.set()
                break
            
            self._monitor_performance()
            time.sleep(MONITOR_INTERVAL_SECONDS)

        self._log("Simulación detenida. Finalizando hilos...")
        # Los hilos daemon se detendrán solos, pero damos un momento
        time.sleep(5) 
        
        self._log("Generando reporte final...")
        self._generate_report()
        self._log("="*50)
        self._log("BENCHMARK FINALIZADO.")
        self._log("="*50)


if __name__ == "__main__":
    benchmark = SystemBenchmark(
        duration_minutes=DURATION_MINUTES,
        battery_threshold=BATTERY_THRESHOLD_STOP
    )
    benchmark.run()

