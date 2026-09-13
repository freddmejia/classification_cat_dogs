# Robustez de la clasificación de gatos y perros en una app móvil KMP

## 1. Conclusión y alcance

Los proyectos examinados combinan controles de captura, gestión de fotogramas, seguimiento de objetos y decisiones temporales. Son mecanismos distintos: algunos mejoran la imagen, otros evitan resultados atrasados y otros reducen los cambios de etiqueta. Una predicción visualmente estable puede seguir siendo incorrecta. La elección debe depender del fallo observado y de una comparación reproducible en el dispositivo.

Para esta app, la recomendación es mantener inicialmente el modelo actual y avanzar en este orden: comprobar equivalencia con el notebook, corregir captura y procesamiento, incorporar rechazo por mala calidad y caducidad de resultados, evaluar suavizado temporal y, si el encuadre lo requiere, añadir detección del animal. No hay evidencia que permita prometer una mejora porcentual concreta sin medir estos cambios sobre imágenes y vídeos propios.

El alcance es Android e iOS con Kotlin Multiplatform. Se distinguen hechos documentados, resultados registrados en el proyecto y propuestas de diseño. No se ha inspeccionado todavía el código de la app móvil ni se han ejecutado pruebas en teléfonos. Este informe no modifica código, configuración, entrenamiento ni modelos.

### Situación del clasificador

El [contrato del modelo](MODEL_CONTRACT.md) describe MobileNetV3Small, entrada RGB de 128 × 128 y una salida sigmoid para perro. Registra estos resultados de Keras:

| Evaluación registrada | Resultado |
|---|---|
| Notebook de entrenamiento, GPU, 2.000 imágenes | Accuracy 94,35%; precisión de perro 95,03%; recall de perro 93,6% |
| Evaluación posterior por clase, CPU | Accuracy 94,40%; recall de gato 95,1%; recall de perro 93,7% |
| Diferencia entre ejecuciones | El contrato atribuye la diferencia a una imagen de perro cercana al umbral |
| TFLite optimizado sobre todo el test | No medido en el contrato |

Estas métricas respaldan una buena clasificación en el conjunto evaluado. No miden específicamente movimiento de cámara, baja luz o estabilidad en vídeo. ImageNet-C e ImageNet-P son precedentes de evaluación que separan precisión normal, resistencia a degradaciones y consistencia ante perturbaciones; sus resultados no deben trasladarse numéricamente a este modelo. [1 · Hendrycks y Dietterich](https://arxiv.org/abs/1903.12261)

Una foto importada y un flujo de cámara requieren diagnósticos diferentes. Si el mismo archivo da resultados distintos entre notebook y móvil, primero hay que comparar modelo, píxeles y salida. El movimiento durante el uso de la cámara no explica por sí mismo esa discrepancia.

## 2. Qué han hecho proyectos y plataformas existentes

| Proyecto o plataforma | Mecanismo observado | Qué aporta | Límite de la evidencia |
|---|---|---|---|
| TensorFlow Lite Camera Demo, ejemplo histórico | Filtro de paso bajo sobre las probabilidades entre fotogramas | Reduce oscilaciones de etiquetas | Demuestra implementación, no una mejora medida para gatos y perros |
| MediaPipe Image Classifier, ejemplos Android/iOS | Modos separados para imagen y vídeo, orientación explícita y resultados asíncronos | Una referencia para integrar cámara e inferencia | Requiere un modelo compatible; no corrige automáticamente la semántica de cualquier TFLite |
| ML Kit Object Detection & Tracking | Detección de regiones y tracking IDs en modo de vídeo | Asocia observaciones al mismo objeto | Sus categorías básicas no incluyen la distinción gato/perro |
| Apple Vision, ejemplo de reconocimiento en directo | Detección con cajas, orientación y posibilidad de iniciar seguimiento | Relaciona la etiqueta con una región de la imagen | Usa un modelo de detección; no equivale al clasificador binario actual |
| Google Camera / HDR+ | Captura en ráfaga, alineación y combinación de imágenes | Mejora fotográfica mediante información de varias capturas | No es una API universal de mejora de los fotogramas de inferencia |
| OpenCV, estudio práctico de enfoque | Comparación de medidas como Tenengrad y Laplaciano | Permite estimar nitidez y seleccionar fotogramas | No proporciona un umbral universal para móviles |

### TensorFlow: suavizar el resultado

La demo filtra las probabilidades antes de mostrarlas mediante `applyFilter()`, con tres etapas y un factor de 0,4. Es un precedente de suavizado temporal; esos valores no están validados para esta app. [2 · Código de ImageClassifier.java](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java)

### MediaPipe: tratar correctamente cada tipo de entrada

Las guías ofrecen modos de imagen, vídeo y flujo en directo. En el modo asíncrono de Android, una llamada recibida mientras el clasificador está ocupado puede descartarse. El ejemplo Android transmite la rotación; el de iOS construye la imagen con su orientación antes de ejecutar la clasificación. Esto permite estudiar por separado el flujo de galería y el de cámara. [3 · Guía Android](https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier/android), [4 · Guía iOS](https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier/ios), [5 · Implementación Android](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/android/app/src/main/java/com/google/mediapipe/examples/imageclassification/ImageClassifierHelper.kt), [6 · Implementación iOS](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/ios/ImageClassifier/Services/ImageClassifierService.swift)

### Google y Apple: localizar antes de interpretar

ML Kit ofrece cajas y un identificador de seguimiento en `STREAM_MODE`. Sus primeras llamadas pueden producir resultados incompletos. La clasificación básica utiliza categorías amplias, por lo que no debe presentarse como un clasificador específico de mascotas. [7 · ML Kit Object Detection & Tracking](https://developers.google.com/ml-kit/vision/object-detection/android)

El ejemplo de Apple reconoce objetos con un modelo de detección y permite utilizar sus observaciones para iniciar `VNTrackObjectRequest`. Esto separa localizar un objeto, seguirlo y asignarle una etiqueta. [8 · Recognizing Objects in Live Capture](https://developer.apple.com/documentation/vision/recognizing-objects-in-live-capture?changes=_1)

### HDR+: mejorar una captura exige gestionar el movimiento

HDR+ combina imágenes de una ráfaga después de alinearlas. El trabajo original usa exposiciones constantes y conserva margen para las altas luces. Es evidencia de una solución de fotografía computacional real, no de que un filtro de contraste aplicado a cualquier fotograma mejore un clasificador. [9 · Hasinoff et al., HDR+](https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/)

Para esta app, la enseñanza práctica es evaluar una captura fotográfica cuando se necesita una respuesta final sobre una foto. Copiar un sistema de fusión de ráfagas sería una ampliación considerable, con alineación, memoria y latencia adicionales.

## 3. Mejoras en la captura del móvil

### Enfoque, exposición y luz

CameraX permite controlar enfoque, medición y compensación de exposición. Las regiones de medición pueden intervenir en AF, AE y AWB: enfoque, exposición y balance de blancos. Conviene comprobar los controles realmente disponibles en la cámara elegida y su estado al reabrir la sesión. [10 · Configuración de CameraX](https://developer.android.com/media/camera/camerax/configuration)

En iOS, AVFoundation permite configurar modos de enfoque y exposición mediante `AVCaptureDevice`; la configuración requiere bloquear el dispositivo durante los cambios. El modo y las capacidades deben comprobarse en el dispositivo concreto. [11 · AVCaptureDevice](https://developer.apple.com/documentation/avfoundation/avcapturedevice)

**Propuesta para la app:** empezar con controles automáticos apropiados para una escena cambiante y permitir dirigir el enfoque al animal. Si se dispone de señales de ajuste de enfoque o exposición, utilizarlas como información de calidad. No mantener un bloqueo de enfoque indefinido mientras el animal cambia de distancia.

El movimiento del teléfono y el del animal pueden producir problemas distintos. Compensar la sacudida de la cámara no congela al animal. Para evaluar exposición, comparar imágenes con buena luz, interior y poca luz, anotando también el ruido: acortar la exposición puede reducir arrastre, pero sacrificar luz. No fijar una velocidad de obturación idéntica para todos los teléfonos sin comprobar resultados.

### Estabilización óptica y electrónica

Android incorporó un modo de estabilización de previsualización en Android 13; CameraX 1.4 añadió acceso a esta función. La documentación explica su aplicación a flujos no RAW y exige consultar soporte. No basta con un vídeo visualmente estable: hay que comprobar las dimensiones, el encuadre y el flujo que realmente consume el analizador. [12 · CameraX 1.4, estabilización](https://android-developers.googleblog.com/2024/12/whats-new-in-camerax-140-and-jetpack-compose-support.html)

En iOS, `preferredVideoStabilizationMode` expresa una preferencia de la conexión. El modo efectivo se consulta con `activeVideoStabilizationMode`; un modo no disponible puede resultar en estabilización desactivada. Apple advierte que puede añadir latencia y consumo de memoria. [13 · Estabilización en AVFoundation](https://developer.apple.com/documentation/avfoundation/avcaptureconnection/preferredvideostabilizationmode?changes=latest_m_3&language=objc)

**Decisión propuesta:** comparar estabilización activada y desactivada en la misma escena, midiendo precisión, retraso y pérdida de encuadre. Activarla por defecto solo donde el balance sea favorable. No atribuirle capacidad para recuperar detalles ya perdidos por desenfoque.

### HDR, modo nocturno y efectos

La API actual `ExtensionSessionConfig` de CameraX indica expresamente que las sesiones de extensiones no admiten `ImageAnalysis`. Esto limita propuestas como activar una extensión HDR o nocturna y seguir utilizando sin cambios el mismo analizador en directo. No significa que toda forma de HDR sea incompatible; significa que esa integración concreta no debe darse por disponible. [14 · CameraX ExtensionSessionConfig](https://developer.android.com/reference/androidx/camera/extensions/ExtensionSessionConfig)

**Recomendación:** mantener inicialmente una conversión reproducible a RGB compatible con el modelo. Evaluar mejoras fotográficas en una ruta de captura separada. No introducir embellecimiento, saturación, contraste agresivo o superresolución como una supuesta mejora automática de precisión.

## 4. Gestión del vídeo y de la antigüedad del resultado

CameraX documenta `STRATEGY_KEEP_ONLY_LATEST` para evitar una cola creciente. También exige liberar `ImageProxy` tras terminar de usarlo. Debe cerrarse ese contenedor, no directamente la imagen compartida que envuelve. [15 · CameraX ImageAnalysis](https://developer.android.com/media/camera/camerax/analyze)

En iOS, la nota TN2445 recomienda `alwaysDiscardsLateVideoFrames` para el procesamiento en tiempo real. Describe una cola de longitud uno, el diagnóstico de fotogramas descartados y la posibilidad de reducir la frecuencia cuando el procesamiento es demasiado lento. Es una nota de 2017, útil como referencia del mecanismo, no una prueba de rendimiento en móviles actuales. [16 · Apple TN2445](https://developer.apple.com/library/archive/technotes/tn2445/_index.html)

**Especificación propuesta:** cada resultado debe conservar el identificador de la captura y su marca temporal. La interfaz debe rechazar resultados de una sesión anterior, de una foto que ya no está seleccionada o de un fotograma demasiado antiguo. Esa caducidad debe basarse en tiempo transcurrido, no únicamente en cuántas inferencias se han ejecutado.

Como punto inicial de experimentación, se puede mantener una vista fluida y clasificar solo 5–10 veces por segundo. Es una propuesta para medir, no una recomendación universal de los proveedores. El objetivo es una respuesta reciente y correcta, sin acumulación ni sobrecalentamiento; ejecutar más inferencias no garantiza mejor experiencia.

## 5. Seleccionar imágenes útiles antes de clasificar

El estudio práctico publicado por OpenCV compara medidas de enfoque, entre ellas Tenengrad, varianza del Laplaciano y combinaciones de Sobel con varianza. Sirven para estimar la presencia de detalle y seleccionar fotogramas nítidos. El artículo no establece un umbral válido para todas las escenas, cámaras y resoluciones. [17 · OpenCV, medidas de enfoque](https://opencv.org/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/)

**Diseño propuesto:** calcular calidad a una resolución fija y, cuando exista una región del animal, sobre esa región. Un fondo texturizado no debería hacer que se acepte una mascota borrosa. El umbral debe calibrarse con ejemplos etiquetados como utilizables o inutilizables para esta tarea; evitar copiar números de un tutorial.

Además de nitidez, conviene evaluar oscuridad, zonas saturadas, tamaño visible del animal y recortes parciales. Cada rechazo necesita una causa concreta: por ejemplo, acercarse, mejorar iluminación o esperar al enfoque. Un único indicador de nitidez no debe decidir todos esos casos.

Cuando se necesite una fotografía definitiva, una opción es escoger el fotograma reciente más nítido dentro de una ventana breve. Hay que limitar también su antigüedad: elegir una imagen excelente de hace varios segundos no representa necesariamente lo que se está mirando ahora.

Un giroscopio puede aportar una señal adicional de movimiento del teléfono, pero no observa el movimiento del animal. Se propone dejar esa combinación de sensores para una segunda fase, si el control visual de calidad resulta insuficiente.

## 6. Estabilidad temporal sin esconder errores

La propuesta para este clasificador es suavizar el valor bruto de perro, conservando también ese valor para diagnóstico. Una media móvil exponencial puede reducir variaciones entre imágenes consecutivas. Su ventana efectiva debería expresarse en tiempo, para que un teléfono lento no retenga evidencia durante mucho más tiempo que uno rápido.

Los filtros de esa demo comienzan en cero. **Inferencia para este modelo:** como cero significa gato, copiar esa inicialización podría sesgar las primeras etiquetas. Se propone comenzar con la primera observación válida y esperar evidencia suficiente. No se ha comprobado que la app actual tenga ese error. [2 · TensorFlow Camera Demo](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java)

### Política propuesta para la interfaz

| Estado conceptual | Condición | Comportamiento |
|---|---|---|
| Esperando imagen útil | Calidad insuficiente o captura en ajuste | Orientar sobre el problema; no inventar una etiqueta |
| Evaluando | Pocas observaciones recientes válidas | Mostrar que la evaluación continúa |
| Resultado estable | Evidencia reciente suficientemente consistente | Mostrar gato o perro |
| No concluyente | Evidencia contradictoria o insuficiente | Abstenerse de confirmar |
| Resultado caducado | Ya no hay evidencia reciente del mismo objeto | Retirar la etiqueta anterior |

Se puede añadir histéresis: exigir más evidencia para cambiar una etiqueta confirmada que para mantenerla. Los límites de esa política deben validarse; no se propone cambiar silenciosamente el umbral de 0,5 del contrato del modelo. El valor bruto y la decisión por imagen siguen disponibles, mientras la interfaz aplica una política temporal versionada.

Una ventana de 300–500 ms es un punto de partida experimental, no un ajuste final. Debe reiniciarse al cambiar de foto, cámara o sesión; también al cambiar de objeto seguido o perderlo durante demasiado tiempo. Las imágenes rechazadas no aportan votos, y la última predicción no debe conservarse indefinidamente mientras la cámara apunta a otra escena.

Las redes pueden estar mal calibradas: un score alto no implica automáticamente esa misma probabilidad de acierto. La calibración es una evaluación distinta de la accuracy. [18 · Guo et al., calibración](https://proceedings.mlr.press/v70/guo17a.html)

Por tanto, un umbral alto tampoco demuestra que exista un animal. El clasificador actual siempre emite un valor para perro frente a gato, incluso ante entradas ajenas a esas clases. Para tratar escenas vacías o sin mascotas, hace falta evaluar explícitamente esa condición, mediante detección u otra estrategia validada.

## 7. Detección y seguimiento: opciones disponibles

### Conservar el modelo y localizar primero al animal

La ruta propuesta es detectar una región, comprobar que es utilizable, recortarla con margen y ejecutar el clasificador sobre ella. Si aparecen varios animales, cada uno necesita identidad y estado propios, o debe existir una selección explícita. Mezclar scores de animales distintos produciría una estabilidad engañosa.

ML Kit presenta precisamente la localización previa como uno de los usos de sus clasificadores personalizados. Sin embargo, exige compatibilidad de tensores y metadatos; la entrada float32 necesita `NormalizationOptions`. El archivo TFLite actual no debe tratarse como intercambiable con cualquier ejemplo: tiene una sola salida para perro y no dos scores explícitos. Habría que comprobar también cómo la API interpretaría esa salida y cómo conservar el rango 0–255 sin normalizar dos veces. [19 · Compatibilidad de modelos con ML Kit](https://developers.google.com/ml-kit/custom-models)

**Recomendación:** si se prueba detección previa, comenzar manteniendo la inferencia actual como componente separado. Validar que el recorte ayuda: entrenamiento y referencia usan la imagen completa redimensionada, de modo que introducir un recorte cambia la distribución de entrada.

### Opciones específicas

| Opción | Uso razonable | Precaución |
|---|---|---|
| Detector básico de ML Kit + clasificador actual | Obtener regiones e identidad en vídeo | Medir si sus regiones sirven para mascotas; sus categorías básicas no resuelven gato/perro |
| MediaPipe Object Detector con un modelo que incluya mascotas | Detectar posición y categoría, o generar regiones | Verificar etiquetas y presupuesto de cómputo del modelo concreto |
| Apple Vision Animal Recognition | Comparador nativo y posible localizador en iOS | Es otro modelo; sus resultados no establecen equivalencia con TFLite |
| Clasificador actual sin detector, con guía de encuadre | Primera versión con un animal claramente visible | No sabe localizar ni separar varios animales |

MediaPipe documenta modelos de detección, entre ellos EfficientDet-Lite0 entrenado sobre COCO, y ejemplos para varias plataformas. La detección añade un componente distinto a la clasificación actual. No se presupone que incluya seguimiento de identidad por el mero hecho de ejecutarse sobre vídeo. [20 · MediaPipe Object Detector](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector)

Apple expone `VNRecognizeAnimalsRequest` y permite consultar sus identificadores soportados. Su sesión WWDC23 explica que Vision ya dispone de reconocimiento de gatos y perros. Es una opción concreta para experimentar sin entrenar un detector desde cero, comprobando disponibilidad en el sistema objetivo. [21 · VNRecognizeAnimalsRequest](https://developer.apple.com/documentation/vision/vnrecognizeanimalsrequest), [22 · Apple WWDC23](https://developer.apple.com/videos/play/wwdc2023/10045/)

**Decisión propuesta:** usar estas alternativas primero como comparación controlada. Si el propósito es mantener un comportamiento equivalente entre Android e iOS, utilizar el mismo modelo en ambas plataformas facilita la comparación. Adoptar clasificadores distintos por plataforma exigiría objetivos de calidad y pruebas independientes.

## 8. Reparto de responsabilidades en KMP

Kotlin Multiplatform permite combinar lógica común con implementaciones específicas e interoperar con APIs nativas. No requiere que los controles de cámara de Android e iOS tengan la misma implementación. [23 · APIs específicas de plataforma en KMP](https://kotlinlang.org/docs/multiplatform/multiplatform-connect-to-apis.html)

La siguiente organización es una propuesta para esta app, no una descripción de su código actual:

| Responsabilidad | Android | iOS | Lógica compartida |
|---|---|---|---|
| Captura y controles | CameraX y capacidades del dispositivo | AVFoundation y capacidades del dispositivo | Estado de sesión y configuración deseada |
| Orientación y píxeles | ImageProxy y conversión verificada a RGB | CVPixelBuffer y conversión verificada a RGB | Contrato de dimensiones, rango y orden |
| Calidad visual | Procesamiento local eficiente | Procesamiento local eficiente | Criterios de aceptación y causas de rechazo |
| Inferencia | Runtime compatible con el TFLite elegido | Runtime compatible con el artefacto elegido | Identidad del modelo y significado de la salida |
| Detección opcional | Detector elegido y probado | Detector elegido y probado | Reglas de selección de objeto |
| Decisión | Entregar score y datos de captura | Entregar score y datos de captura | Suavizado, histéresis, caducidad y estado de UI |
| Medición | Tiempos y señales nativas | Tiempos y señales nativas | Formato de resultados y análisis comparativo |

Evitar transportar innecesariamente imágenes completas a través de varias capas solo para compartir código. Es preferible compartir decisiones y datos pequeños: identificador, tiempo, score, calidad y región. Las marcas temporales deben tener una referencia consistente dentro de cada ejecución.

Apple señala que BGRA no es un formato nativo de captura y puede requerir conversión y más memoria que otros formatos. Por ello, la ruta más sencilla para depurar no tiene por qué ser la más eficiente para producción. Primero hay que verificar colores y equivalencia; después, medir conversiones y copias. [24 · AVCaptureVideoDataOutput](https://developer.apple.com/documentation/avfoundation/avcapturevideodataoutput?changes=__4)

La ruta de foto importada debe mantenerse determinista y sin historial temporal. El resultado de una foto nueva no debe depender de lo que la cámara clasificó anteriormente. Una captura definitiva también debería presentar el resultado correspondiente a esa captura, sin heredar automáticamente la etiqueta suavizada de la vista previa.

## 9. Plan de validación en el móvil

### Fase A: equivalencia con el modelo publicado

Usar inicialmente el TFLite normal sobre CPU como referencia controlada. Comprobar el checksum del artefacto realmente instalado, no solo del archivo que aparece en el repositorio. Ejecutar las imágenes de referencia sin recorte y comparar valor bruto y etiqueta. El contrato registra:

| Imagen | TFLite normal | TFLite optimizado | Etiqueta |
|---|---:|---:|---|
| dog.png | 0,99254 | 0,99545 | Perro |
| cat.jpg | 0,00225 | 0,00266 | Gato |

La entrada debe ser `[1,128,128,3]`, float32 RGB y valores 0–255. La salida es un único valor: mayor que 0,5 significa perro. La tolerancia registrada para las referencias en app es ±0,02. Estos son resultados previos del [contrato](MODEL_CONTRACT.md); no son inferencias nuevas realizadas durante esta investigación.

Si hay discrepancia, comparar la imagen transformada y el tensor final antes de cambiar el entrenamiento. Revisar especialmente normalización duplicada, canales, orientación, lectura del buffer, modelo desactualizado y traducción de score a etiqueta. El [contexto de inspección móvil](../MOBILE_INFERENCE_CONTEXT.md) detalla ese diagnóstico.

### Fase B: colección de situaciones reales

Se propone un conjunto inicial de clips cortos, por ejemplo de 5–10 segundos, con gatos y perros variados. Incluir teléfono estable, sacudida de mano, movimiento del animal, cambio de distancia, luz interior, poca luz, fondos complejos, animal pequeño, oclusión, entrada y salida de escena y ausencia de mascota. Añadir cambios rápidos entre una escena con gato y otra con perro.

Separar calibración y evaluación por animal o sesión de captura; fotogramas adyacentes no deben repartirse entre ambos grupos como si fueran observaciones independientes. Mantener vídeos sin animal para medir respuestas forzadas y probar sesiones suficientemente largas para observar degradación térmica.

En una primera ronda, elegir teléfonos Android de distintas capacidades y al menos dos generaciones de iPhone si están disponibles. Es una propuesta de cobertura, no una muestra suficiente para certificar todos los dispositivos.

### Fase C: comparar medidas de una en una

| Variante experimental | Cambio respecto al control | Pregunta |
|---|---|---|
| A | Integración correcta sin mejoras adicionales | ¿Cuál es el rendimiento real de partida? |
| B | Gestión de fotogramas y caducidad | ¿Desaparecen resultados atrasados? |
| C | Rechazo por calidad | ¿Bajan los errores sin abstenerse excesivamente? |
| D | Suavizado y política temporal | ¿Se estabiliza la etiqueta sin tardar demasiado en cambiar? |
| E | Estabilización de captura | ¿Mejora el resultado en dispositivos compatibles? |
| F | Detección y recorte | ¿Ayuda cuando el animal ocupa poca imagen? |
| G | Modelo optimizado o aceleración | ¿Se conserva calidad con menor coste? |

Cuando se evalúan decisiones sobre píxeles, pueden reutilizarse los mismos clips. Cuando se modifica enfoque, exposición o estabilización del sensor, hace falta una captura nueva controlada; reproducir un vídeo ya grabado no permite medir esos efectos.

### Métricas que debe recoger la app

| Métrica | Qué revela |
|---|---|
| Matriz de confusión y recall por clase | Si el error perjudica especialmente a gatos o perros |
| Acierto entre resultados confirmados | Calidad de las respuestas que sí se presentan |
| Cobertura o tasa de respuesta | Cuánto tiempo se abstiene la app |
| Falsas confirmaciones sin animal | Si fuerza gato/perro ante escenas no válidas |
| Cambios incorrectos de etiqueta por minuto | Inestabilidad visible |
| Tiempo hasta confirmar correctamente | Rapidez de respuesta útil |
| Tiempo para retirar una etiqueta al salir el animal | Persistencia de resultados antiguos |
| Antigüedad del fotograma al mostrar el resultado, p50/p95 | Retraso real de extremo a extremo |
| Inferencia, conversión y captura por separado | Origen del coste |
| Comportamiento tras uso sostenido | Efecto de temperatura y carga |

No aprobar una medida solo porque sube el acierto entre respuestas aceptadas. Si para lograrlo deja casi todas las escenas sin respuesta, el producto puede empeorar. Tampoco aprobarla solo porque disminuyen los cambios de etiqueta: mantener siempre gato produciría una interfaz estable e incorrecta.

Los umbrales de calidad y tiempo se ajustan con el conjunto de calibración. Después se evalúan una sola vez sobre sesiones reservadas y se documentan los resultados por plataforma. Los objetivos finales de latencia y abstención deben fijarse según la experiencia buscada, no extraerse de una demo genérica.

## 10. Prioridades y decisiones propuestas

| Prioridad | Acción | Cambio de modelo |
|---|---|---|
| P0 | Resolver la discrepancia con la foto importada y verificar entrada/salida | No |
| P1 | Auditar enfoque, exposición, orientación, colas y asociación entre captura y resultado | No |
| P1 | Incorporar criterios medidos de calidad y resultado caducado | No |
| P2 | Probar suavizado temporal con reinicio correcto e histéresis | No; versionar la política de la app |
| P2 | Medir estabilización en dispositivos compatibles | No |
| P3 | Comparar detector + recorte cuando el encuadre sea una causa importante | Añade componente y requiere validar el nuevo procesamiento |
| P3 | Evaluar captura fotográfica para una confirmación final | No necesariamente |
| Posterior | Reentrenar con condiciones móviles si persisten errores sistemáticos | Sí |

El comportamiento propuesto para una primera versión robusta es: capturar con una configuración adecuada, procesar una imagen reciente, comprobar calidad, preparar exactamente la entrada esperada, inferir, evaluar evidencia temporal del mismo objeto y mostrar una respuesta que pueda caducar. La detección previa se añade cuando el conjunto de pruebas demuestre que la localización es necesaria.

No se recomienda comenzar por deblurring neuronal, superresolución, fusión de varios clasificadores ni entrenamiento nuevo. Podrían estudiarse después con un fallo concreto y un criterio de mejora; incorporarlos ahora complicaría distinguir una mejora real de un error de integración oculto.

## Fuentes

Consulta de fuentes web: 13 de septiembre de 2026. Las páginas de documentación y las ramas `main`/`master` son referencias evolutivas; la implementación deberá contrastarlas con las versiones del proyecto. Los ejemplos históricos se utilizan como precedentes de diseño, no como recomendación de dependencias antiguas.

1. Dan Hendrycks y Thomas Dietterich. [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), ICLR 2019. Evaluación de degradaciones y perturbaciones.
2. TensorFlow. [ImageClassifier.java, TFLite Camera Demo](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java), ejemplo histórico, rama consultada `master`. Filtro temporal e inicialización.
3. Google AI Edge. [Image classification guide for Android](https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier/android), actualización visible 17-08-2026. Modos de ejecución y descarte durante ocupación.
4. Google AI Edge. [Image classification guide for iOS](https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier/ios), documentación vigente consultada. Ejemplo de cámara y biblioteca multimedia.
5. Google AI Edge. [ImageClassifierHelper.kt](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/android/app/src/main/java/com/google/mediapipe/examples/imageclassification/ImageClassifierHelper.kt), rama `main`. Orientación explícita en Android.
6. Google AI Edge. [ImageClassifierService.swift](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/ios/ImageClassifier/Services/ImageClassifierService.swift), rama `main`. Orientación y clasificación asíncrona en iOS.
7. Google ML Kit. [Detect and track objects on Android](https://developers.google.com/ml-kit/vision/object-detection/android), documentación vigente consultada. Tracking IDs, resultados iniciales y categorías básicas.
8. Apple. [Recognizing Objects in Live Capture](https://developer.apple.com/documentation/vision/recognizing-objects-in-live-capture?changes=_1), código de ejemplo. Detección y posible seguimiento de objetos.
9. Samuel W. Hasinoff et al., Google Research. [Burst photography for high dynamic range and low-light imaging on mobile cameras](https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/), ACM Transactions on Graphics, 2016. HDR+ y ráfagas.
10. Android Developers. [CameraX configuration options](https://developer.android.com/media/camera/camerax/configuration), documentación vigente consultada. Enfoque, medición y exposición.
11. Apple. [AVCaptureDevice](https://developer.apple.com/documentation/avfoundation/avcapturedevice), documentación vigente consultada. Configuración nativa de captura.
12. Scott Nien, Android Developers Blog. [What's new in CameraX 1.4.0](https://android-developers.googleblog.com/2024/12/whats-new-in-camerax-140-and-jetpack-compose-support.html), 17-12-2024. Estabilización y capacidades.
13. Apple. [preferredVideoStabilizationMode](https://developer.apple.com/documentation/avfoundation/avcaptureconnection/preferredvideostabilizationmode?changes=latest_m_3&language=objc), documentación vigente consultada. Modo efectivo, latencia y memoria.
14. Android Developers. [ExtensionSessionConfig](https://developer.android.com/reference/androidx/camera/extensions/ExtensionSessionConfig), API añadida en CameraX 1.6.0. Ausencia de soporte para ImageAnalysis en sesiones de extensiones.
15. Android Developers. [Image analysis](https://developer.android.com/media/camera/camerax/analyze), documentación vigente consultada. Backpressure, orientación y liberación de buffers.
16. Apple. [TN2445: Handling Frame Drops with AVCaptureVideoDataOutput](https://developer.apple.com/library/archive/technotes/tn2445/_index.html), 12-07-2017. Descarte de fotogramas y diagnóstico.
17. Moukthika, OpenCV. [Autofocus using OpenCV: A Comparative Study of Focus Measures for Sharpness Assessment](https://opencv.org/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/), 11-03-2025. Comparación práctica de medidas de enfoque.
18. Chuan Guo, Geoff Pleiss, Yu Sun y Kilian Q. Weinberger. [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017. Diferencia entre score y probabilidad calibrada.
19. Google ML Kit. [Custom models with ML Kit](https://developers.google.com/ml-kit/custom-models), actualización visible 10-09-2026. Requisitos de tensores y metadatos.
20. Google AI Edge. [Object detection task guide](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector), documentación vigente consultada. Detección, modelos y plataformas.
21. Apple. [VNRecognizeAnimalsRequest](https://developer.apple.com/documentation/vision/vnrecognizeanimalsrequest), documentación vigente consultada. Reconocimiento de animales e identificadores soportados.
22. Nadia Zouba, Apple. [Detect animal poses in Vision](https://developer.apple.com/videos/play/wwdc2023/10045/), WWDC23. Reconocimiento existente de gatos y perros y capacidades relacionadas.
23. JetBrains. [Use platform-specific APIs](https://kotlinlang.org/docs/multiplatform/multiplatform-connect-to-apis.html), documentación vigente consultada. Integración nativa desde KMP.
24. Apple. [AVCaptureVideoDataOutput](https://developer.apple.com/documentation/avfoundation/avcapturevideodataoutput?changes=__4), documentación vigente consultada. Formatos de captura y coste de BGRA.
25. Proyecto local. [MODEL_CONTRACT.md](MODEL_CONTRACT.md), contrato 1.0.0, consultado 13-09-2026. Métricas registradas, formato de entrada/salida y referencias. [MOBILE_INFERENCE_CONTEXT.md](../MOBILE_INFERENCE_CONTEXT.md) conserva el contexto de inspección de la app.
