```markdown
# Proyecto Integrado V - COLCAP Tracker

Este proyecto tiene como objetivo automatizar la recolección continua de datos históricos del índice bursátil **COLCAP**, principal indicador del mercado accionario colombiano. Los datos se obtienen de Yahoo Finanzas y se almacenan en un archivo CSV, manteniendo su trazabilidad en un entorno de control de versiones con GitHub.

## 📌 Características

- Descarga diaria automática de datos usando GitHub Actions.
- Persistencia histórica sin pérdida de registros anteriores.
- Registro de ejecución con sistema de logging.
- Implementado con programación orientada a objetos (OOP).
- Documentación del proyecto en formato APA.

## ⚙️ Tecnologías utilizadas

- Python 3
- yfinance
- pandas
- GitHub Actions

## 📁 Estructura del proyecto

```
```
proyecto\_integrado\_v\_colcap-tracker/
├── .github/workflows/update\_data.yml
├── docs/report\_entrega1.pdf
├── src/
│   ├── collector.py
│   ├── logger.py
│   └── static/historical.csv
└── README.md
```

```

## 📈 Indicador monitoreado

- **Índice COLCAP** (Símbolo en Yahoo Finanzas: `^737809-COP-STRD`)  
- [Ver en Yahoo Finanzas](https://es.finance.yahoo.com/quote/%5E737809-COP-STRD/)

## 📄 Licencia

Este proyecto es únicamente con fines educativos para la asignatura Proyecto Integrado V.
```