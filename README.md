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
├── configs/                  # Configuraciones e hiperparametros de cada modelo
├── src/                      # Código fuente de los clasificadores
│   ├── trainer.py            # Defino clase Trainer para el entrenamiento de modelos
│   ├── dataload.py            # dataloaders para entrenamiento, validación y prueba a partir de audio
│   ├── dataset.py            # Cargar y procesar archivos para modelos
│   ├── model.py              # Definicion dos clases de modelos (NeuralNet y NeuralNet_mfcc)
│   └── eval.py               # Defino clase Evaluator para evlauar predicciones
├── tests.py                    # Pruebas unitarias y de integración
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
