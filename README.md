```markdown
# Proyecto Integrado V - Grupo Aval Tracker (AVAL)

Este proyecto tiene como objetivo automatizar la recolección continua de datos históricos del 
**Grupo Aval (AVAL)**, una de las principales entidades financieras de Colombia. Los datos se 
obtienen desde **Yahoo Finanzas**, se almacenan en formato `.csv` y se actualizan automáticamente 
mediante **GitHub Actions**, manteniendo la trazabilidad y persistencia del histórico.

---

## 📌 Características

- 🔄 Automatización diaria con GitHub Actions
- 📊 Almacenamiento histórico en `CSV`
- 🔍 Logs de ejecución para trazabilidad
- 🧱 Implementación con **Programación Orientada a Objetos (OOP)**
- 🧪 Recolector de datos con `yfinance` y `pandas`

---

## ⚙️ Tecnologías utilizadas

- Python 3.10
- [yfinance](https://pypi.org/project/yfinance/)
- pandas
- logging
- GitHub Actions

---

## 📈 Indicador económico

- **Activo**: Grupo Aval Acciones y Valores S.A.
- **Símbolo**: `AVAL`
- [🔗 Ver en Yahoo Finanzas](https://es-us.finanzas.yahoo.com/quote/AVAL/)

---

## 📁 Estructura del repositorio

```
```
proyecto\_integrado\_v\_aval/
├── .github/
│   └── workflows/
│       └── update\_data.yml      # Flujo automático de actualización
│
├── docs/
│   └── report\_entrega1.pdf      # Informe académico en formato APA
│
├── src/
│   ├── collector.py             # Descarga y persistencia de datos
│   ├── logger.py                # Configuración de logs
│   └── static/
│       └── historical.csv       # Datos históricos de AVAL
│
├── README.md
└── .gitignore

```

---

## 🚀 Instrucciones de uso

1. Instala dependencias:
   ```bash
   pip install yfinance pandas
````

2. Ejecuta el colector localmente:

   ```bash
   python src/collector.py
   ```

3. (Opcional) Configura y ejecuta GitHub Actions para automatización.

---

## 📄 Licencia

Este proyecto es de uso educativo y forma parte de la asignatura **Proyecto Integrado V**, bajo la línea de énfasis en automatización y análisis económico.

---

```
