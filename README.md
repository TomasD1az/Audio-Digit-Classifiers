# Clasificadores de Números en Audio

Este repositorio contiene varios clasificadores de audio capaces de reconocer números del 0 al 9. El sistema toma grabaciones de audio como entrada y devuelve el número detectado en la grabación.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Descripción

El objetivo de este proyecto es desarrollar y probar diferentes modelos de clasificación de audio para reconocer números hablados del 0 al 9. Estos modelos pueden ser útiles en aplicaciones de reconocimiento de voz y sistemas interactivos que requieren entender comandos numéricos.

## Instalación

Para utilizar los clasificadores, sigue estos pasos:

1. Clona este repositorio:

    ```bash
    git clone https://github.com/TomasD1az/Clasificador_An_III.git
    cd Clasificador_An_III
    ```

2. Crea un entorno virtual (opcional pero recomendado):

    ```bash
    python -m venv env
    source env/bin/activate  # En Windows usa `env\Scripts\activate`
    ```

3. Instala las dependencias necesarias:

    ```bash
    pip install -r requirements.txt
    ```
## Estructura del Repositorio

```plaintext
/
├── data/                     # Carpeta para almacenar los datos de audio
├── models/                   # Modelos entrenados
├── notebooks/                # Notebooks de Jupyter con análisis y experimentos
├── src/                      # Código fuente de los clasificadores
│   ├── preprocesamiento.py   # Funciones de preprocesamiento de audio
│   ├── modelo.py             # Definición y entrenamiento de modelos
│   └── clasificador.py       # Funciones de carga de modelos y predicciones
├── tests/                    # Pruebas unitarias y de integración
├── requirements.txt          # Lista de dependencias del proyecto
└── README.md                 # Este archivo
```

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas contribuir, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b mejora-nueva`).
3. Haz los cambios necesarios y comete tus cambios (`git commit -am 'Añadir nueva mejora'`).
4. Sube tus cambios a tu rama (`git push origin mejora-nueva`).
5. Crea un Pull Request.

## Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo LICENSE
