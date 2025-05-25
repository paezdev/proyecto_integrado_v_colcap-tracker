from setuptools import setup, find_packages

setup(
    name="proyecto_integrado_v_aval_tracker",
    version="0.0.2",
    author="Jean Carlos Páez Ramírez",
    author_email="",
    description="Paquete para análisis y predicción de acciones del grupo AVAL usando datos de Yahoo Finance",
    packages=find_packages(where="src"),  # Busca paquetes en la carpeta src
    package_dir={"": "src"},              # Indica que los paquetes están en src
    install_requires=[
        "pandas>=2.2.3",
        "numpy",
        "scikit-learn",
        "streamlit",
        "plotly",
        "joblib",
        "yfinance>=0.1.64",
        "matplotlib",
        "seaborn",
        "statsmodels"
    ],
    python_requires=">=3.8",  # Mejor usar 3.8+ para compatibilidad
    include_package_data=True,
)