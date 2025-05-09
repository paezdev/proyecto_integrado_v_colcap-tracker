## Proyecto Integrado V - Grupo Aval Tracker (AVAL)

Este proyecto tiene como objetivo automatizar la recolección continua de datos históricos del **Grupo Aval (AVAL)**, 
una de las principales entidades financieras de Colombia. Los datos se obtienen desde **Yahoo Finanzas**, se almacenan 
en formato `.csv` y se actualizan automáticamente mediante **GitHub Actions**, manteniendo la trazabilidad y 
persistencia del histórico.

---

## 📌 Características

* 🔄 **Automatización diaria con GitHub Actions**: Los datos se actualizan automáticamente cada día a las 12:00 UTC.
* 📊 **Almacenamiento histórico en `CSV`**: Los datos se mantienen en formato CSV para facilitar su análisis.
* 🔍 **Logs de ejecución para trazabilidad**: Se guarda un archivo `log_data.csv` con los registros de cada ejecución.
* 🧱 **Implementación con Programación Orientada a Objetos (OOP)**: El código se organiza utilizando principios de OOP.
* 🧪 **Recolector de datos con `yfinance` y `pandas`**: El colector de datos descarga los datos históricos de Yahoo Finanzas y los guarda en un archivo CSV.
* 🧾 **Logger personalizado en CSV**: Se usa un logger en formato CSV para almacenar logs de ejecución estructurados.
* 📦 **Distribución del paquete con `setup.py`**: Estructura preparada para instalación como paquete Python local o remoto.

---

## ⚙️ Tecnologías utilizadas

* Python 3.10
* [yfinance](https://pypi.org/project/yfinance/)
* pandas
* logging
* GitHub Actions

---

## 📈 Indicador económico

* **Activo**: Grupo Aval Acciones y Valores S.A.
* **Símbolo**: `AVAL`
* [🔗 Ver en Yahoo Finanzas](https://es-us.finanzas.yahoo.com/quote/AVAL/)

---

## 📁 Estructura del repositorio

```
proyecto_integrado_v_aval/
├── .github/
│   └── workflows/
│       └── update_data.yml          # Flujo automático de actualización con GitHub Actions
│
├── docs/
│   └── report_entrega1.pdf          # Informe académico en formato APA
│
├── src/
│   ├── collector.py                 # Descarga y persistencia de datos
│   ├── logger.py                    # Configuración base del logger
│   ├── csv_logger.py                # Manejador personalizado de logs en formato CSV
│   ├── static/
│   │   └── historical.csv           # Datos históricos de AVAL
│   └── logs/
│       └── log_data.csv             # Logs de cada ejecución en formato CSV
│
├── setup.py                         # Script de configuración para instalación como paquete
├── README.md
└── .gitignore
```

---

## 🚀 Instrucciones de uso

1. **Instala dependencias**:

   ```bash
   pip install yfinance pandas
   ```

2. **Ejecuta el colector localmente**:

   ```bash
   python src/collector.py
   ```

3. **Automatización con GitHub Actions**:
   GitHub Actions ejecuta automáticamente el flujo en `.github/workflows/update_data.yml` todos los días a las 12:00 UTC.
   Los datos se actualizan, se almacenan en `historical.csv`, y los logs quedan en `src/logs/log_data.csv`.

---

## 📄 Licencia

Este proyecto es de uso educativo y forma parte de la asignatura **Proyecto Integrado V**, 
bajo la línea de énfasis en automatización y análisis económico.

---

### Explicación adicional de archivos clave

* **`setup.py`**: Archivo que permite instalar el proyecto como un paquete Python. Facilita su distribución, 
reutilización y empaquetado. Ideal si deseas instalar tu proyecto con `pip install .`.

---