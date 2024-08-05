# IA
 Para poder generar la imagen utilizamos las funciones de la Interfaz de Usuario de ComfyUI
 Por eso para hacer fucionar el script de generacion primero debemos clonar el siguiente repositorio 
 https://github.com/comfyanonymous/ComfyUI

 Una vez hecho esto tenemos que descargar el siguiente modelo de Civitai.com
 https://civitai.com/models/312530/cyberrealistic-xl

 Y guardar el archivo .safetensors en el directorio /ComfyUI/models/checkpoints

 Tambien debemos guardar el archivo blue pepsi can-000005.safetensors en el directorio /ComfyUI/models/loras

 Una vez tengamos eso preparado ya podemos ejecutar el codigo que va a estar encustando a la cola de SQS cada 10 segundos
