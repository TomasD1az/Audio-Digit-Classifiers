# Instrucciones para correr el código

Para entrenar el modelo, desde una terminal escribir:

```
  python train.py "config_name"
```

en donde 'config_name' refiere a alguno de los tres archivos de configuración provistos, uno por tipo de feature de entrada. Entonces, si se quisiera entrenar un modelo cuyo dato de entrada fuera un espectro, uno escribiría:

```
  python train.py "dft_config"
```

Al entrenar el modelo se generarán automaticamente carpetas con el nombre del mismo dentro de la carpeta 'models'. En la subcarpeta creada se almacenarán los pesos del sistema.

Para evaluar un modelo el procedimiento es el mismo, pero el archivo a ejecutar es 'test.py' en vez de 'train.py'. Entonces, retomando el ejemplo del modelo basado en espectros de entrada:

```
  python test.py "dft_config"
```

Al finalizar la ejecución de este archivo, se generará una carpeta con el nombre del modelo dentro de la carpeta 'predictions'. Dentro de esa carpeta se podrá encontrar un archivo con extensión .npz en donde están almacenados los datos de entrada al modelo, las etiquetas objetivo para cada audio y las estimaciones de las etiquetas hechas por el modelo para cada dígito.

Ante cualquier duda, consulten en clase o por mail!
