# APP_SEGMENTACION_ESPECIES

Esta aplicación de Streamlit permite clasificar y agrupar variedades de fruta utilizando reglas y algoritmos de clustering.

## Instalación

```bash
pip install -r requirements.txt
```

### Problemas durante la instalación

Si al instalar `scikit-learn` se muestra un error similar a:

```
OSError: [WinError 32] El proceso no tiene acceso al archivo porque está siendo utilizado por otro proceso
```

prueba estos pasos:

1. Cierra cualquier proceso de Python o `pip` en ejecución y, si es necesario, tu IDE.
2. Elimina la carpeta temporal indicada en el mensaje de error, por ejemplo `C:\Users\usuario\AppData\Local\Temp\pip-unpack-XXXX`.
3. Vuelve a ejecutar la instalación sin utilizar la caché:

   ```bash
   pip install --no-cache-dir scikit-learn
   ```
4. Desactiva temporalmente el antivirus mientras dura la instalación.
5. Si el problema continúa, reinicia la máquina e intenta de nuevo.

## Uso

Ejecute el siguiente comando y abra la URL que se indica en el terminal:

```bash
streamlit run app2.py
```

Al subir un archivo XLSX, la aplicación leerá automáticamente la hoja
**Ev. Cosecha Extenso** para aplicar la clasificación.

