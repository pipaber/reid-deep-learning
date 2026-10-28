# Guion del video explicativo

**Proyecto:** Re-identificación de personas auto-supervisada con ReID-oriented Prior y destilación
maestro–estudiante (Yang et al., 2025)  
**Duración objetivo:** 17 minutos y 30 segundos  
**Límite:** 18 minutos

> Las indicaciones entre corchetes no se narran. Los tiempos son acumulados y
> sirven como referencia. Las diapositivas del apéndice se reservan para
> preguntas. Los valores están en porcentaje.

---

# 1. Metodología — 0:00 a 6:00

## Slide: Portada

**Tiempo: 0:00–0:10**

“Hola. En este video presentamos nuestra implementación de un método de
re-identificación de personas auto-supervisado, basado en DINO con un prior
orientado a ReID, adaptado al dataset Market-1501 del laboratorio.”

[Pasar la portada de sección “Metodología” sin detenerse.]

## Slide: Encontrar a la misma persona en otra cámara

**Tiempo: 0:10–1:00**

“El problema es el siguiente: dada una imagen de consulta de 128 por 64
píxeles, el modelo debe producir un vector y ordenar una galería de imágenes
por similitud coseno, de modo que la misma persona, vista desde otra cámara,
quede arriba.

Entrenamos con 11 043 imágenes de seis cámaras, sin ninguna etiqueta de
identidad. Validamos con 100 consultas y una galería de 1 793 imágenes, y
evaluamos en test con 3 368 consultas y 19 732 imágenes de galería, de 750
personas que no aparecen en entrenamiento. Reportamos mAP y Rank-1, 5 y 10 con
el protocolo de Market-1501, en dos condiciones: consultas originales y
consultas reducidas a un cuarto de resolución.”

## Slide: El dataset: Market-1501

**Tiempo: 1:00–1:40**

“Estas son dieciséis imágenes de entrenamiento con su cámara. Son recortes de
personas en un campus universitario. [Señalar la figura.] Cambian la pose, el
fondo, la iluminación y la escala; varios recortes están desenfocados o
cortados. Las personas de test son distintas de las de entrenamiento, así que
el modelo no puede memorizar identidades: tiene que aprender qué hace
reconocible a una persona.”

## Slide: ¿Por qué es difícil?

**Tiempo: 1:40–2:25**

“Tres razones. Primero, la misma persona cambia de punto de vista, fondo y
escala entre cámaras. Segundo, las consultas de baja resolución son de 32 por
16 píxeles: a la izquierda están las originales y abajo las reducidas; se
pierde el detalle de la ropa y los accesorios. Tercero, personas distintas con
ropa parecida se confunden fácilmente.

Y todo esto sin etiquetas: el modelo tiene que descubrir por sí mismo qué es
invariante.”

## Slide: Punto de partida: DINO v1

**Tiempo: 2:25–3:15**

“El paper parte de DINO. Hay dos redes idénticas: un estudiante, que se
entrena por gradiente, y un maestro, que es una media móvil exponencial de los
pesos del estudiante. Cada imagen se aumenta en dos vistas. El maestro
produce una distribución sobre K prototipos, y el estudiante, viendo la otra
vista, debe imitarla con una cross-entropy.

No hay etiquetas ni pares negativos. Para que no colapse, DINO usa centering,
que resta al maestro una media móvil de sus salidas para que ningún prototipo
domine, y sharpening, una temperatura baja en el maestro para que la
distribución no sea uniforme.”

## Slide: Lo que agrega el paper: ReID-oriented Prior

**Tiempo: 3:15–4:05**

“El núcleo del paper es el ReID-oriented Prior. El maestro ve las dos vistas
limpias, m y n. El estudiante ve una versión degradada de cada una. [Señalar
las columnas.] Primero una transformación de baja calidad, LQ: blur gaussiano
con sigma 1 y motion blur de longitud 3 con un ángulo aleatorio. Después una
máscara aleatoria: el 60 por ciento de los tokens de la imagen se reemplaza
por un token de máscara aprendible.

La idea es simular las dos dificultades de la vigilancia real: baja calidad de
imagen y oclusión. El estudiante tiene que restaurar, en el espacio de
embeddings, lo que el maestro ve completo.”

## Slide: Dos proyectores, dos pérdidas

**Tiempo: 4:05–5:05**

“Sobre el backbone hay un proyector: un MLP de tres capas, normalización L2 y
una capa de prototipos. El paper comparte sus pesos entre el token CLS y los
patches.

Hay dos pérdidas. La de patches, ecuación 1, compara cada token del estudiante,
degradado y enmascarado, con el mismo token del maestro, limpio, dentro de la
misma vista. Es la pérdida de iBOT. La de CLS, ecuación 2, compara el token
global del estudiante en la vista n con el del maestro en la vista m: es la
pérdida de DINO, cruzada entre vistas.

La pérdida total, ecuación 3, suma las de patches de ambas vistas y las de CLS
con un peso lambda de 0.55. La de patches restaura detalle local; la de CLS
alinea la semántica global.”

## Slide: Nuestra ablación: dos ideas de DINOv2

**Tiempo: 5:05–6:00**

“Como ablación tomamos dos mejoras de DINOv2. La primera reemplaza el
centering por Sinkhorn-Knopp: el maestro produce una matriz de imágenes por
prototipos doblemente estocástica, normalizando filas y columnas tres veces,
lo que obliga a repartir el batch por igual entre los prototipos.

La segunda es el regularizador KoLeo: para cada embedding del batch, penaliza
menos log de la distancia a su vecino más cercano. Maximiza la entropía
diferencial, es decir, separa los embeddings. Lo pesamos con 0.1, como en
DINOv2. La hipótesis es que un espacio más uniforme ayude a distinguir
personas parecidas.”

---

# 2. Implementación — 6:00 a 12:00

## Slide: Del paper al laboratorio

**Tiempo: 6:00–7:00**

“Estas son las adaptaciones. El paper pre-entrena con LUPerson, 4 millones de
imágenes; nosotros con las 11 mil del laboratorio. Usamos un ViT-Small con
patch 8 sobre la imagen nativa de 128 por 64, que da los mismos 128 tokens que
el ViT-S/16 del paper a 256 por 128, sin inventar píxeles.

A la transformación de baja calidad le agregamos una reducción por cuatro con
probabilidad 0.5, porque es exactamente la condición de test. Y la diferencia
más importante: el paper siempre hace fine-tuning supervisado con triplet loss
después del pre-entrenamiento. El laboratorio prohíbe usar identidades, así
que evaluamos el pre-entrenamiento en crudo: el embedding es el token CLS del
backbone, normalizado, con similitud coseno. El paper nunca reporta ese
número.”

## Slide: La arquitectura, en 150 líneas

**Tiempo: 7:00–8:00**

“Toda la propuesta es código propio. En model.py, el ViT: una convolución 8
por 8 produce 128 tokens, más el CLS, un token de máscara aprendible y 12
bloques pre-norm de 384 dimensiones; 21 millones de parámetros. El proyector
es un MLP 384, 2048, 2048, 256, normalización L2 y 4096 prototipos con pesos
normalizados.

En prior.py están el blur gaussiano, el motion blur con un ángulo por imagen,
la reducción y la máscara aleatoria. En loss.py, la destilación con centering
o Sinkhorn-Knopp, y KoLeo.

En train.py, cada red tiene un solo proyector que se aplica al token CLS y a
los patches: así se comparten los pesos. Un roll del batch empareja la vista m
del maestro con la n del estudiante, que es la pérdida cruzada. Y el maestro se
actualiza con una interpolación lineal de los pesos después de cada paso.”

## Slide: Cómo entrenamos

**Tiempo: 8:00–8:45**

“Usamos los hiperparámetros de DINO con el learning rate escalado por el
batch: AdamW con 2.5 por 10 a la menos 4, weight decay 0.04, diez épocas de
warmup y decaimiento coseno. 100 épocas de 86 pasos, unos 65 segundos por
época en una RTX A2000 con bf16.

La temperatura del maestro sube de 0.04 a 0.07 en 30 épocas; la del
estudiante es 0.1. El momentum del maestro va de 0.996 a 1. Recortamos el
gradiente a 3 y congelamos los prototipos en la primera época, como DINO.

Cada run cuesta 1.8 horas. Con el plazo del laboratorio priorizamos tres runs
y una cuarta configuración del prior.”

## Slide: Cómo evaluamos

**Tiempo: 8:45–9:25**

“El embedding es el token CLS normalizado y la similitud es coseno. Seguimos
el protocolo de Market-1501: para cada consulta se descartan las imágenes de
galería con la misma identidad y la misma cámara, se ignoran las marcadas como
junk y los distractores cuentan como negativos.

Las consultas de baja resolución, de 32 por 16, se reescalan bicúbicamente a
128 por 64 antes de entrar a la red; el modelo es el mismo. Reportamos maestro
y estudiante por separado.”

## Slide: ¿Qué comparamos?

**Tiempo: 9:25–10:10**

“Cinco configuraciones con el mismo backbone, los mismos hiperparámetros y las
mismas 100 épocas. Un ViT sin entrenar como piso. DINO v1 como modelo base: en
el mismo código apagamos el prior y la pérdida de patches, así que la
comparación aísla exactamente lo que aporta el paper. El método completo del
paper. El paper con Sinkhorn-Knopp y KoLeo, que es nuestra ablación. Y una
cuarta configuración con el prior más suave: máscara del 30 por ciento y sin
reducción por cuatro.”

[Pausa breve. Si hay tiempo, mencionar que los flags están en el README.]

**Tiempo: 10:10–12:00 — reserva (recortar aquí si se pasa de 18:00)**

[Esta reserva absorbe las explicaciones de código si el evaluador pide ver
train.py en pantalla: mostrar el bloque del prior, las dos llamadas al
proyector y el roll de la pérdida cruzada.]

---

# 3. Resultados — 12:00 a 17:30

## Slide: Resultados en test (maestro)

**Tiempo: 12:00–13:00**

“Esta es la tabla de test con el maestro. El ViT sin entrenar da
@@random.teacher.original.mAP@@ de mAP y @@random.teacher.original.R1@@ de Rank-1: ese es el piso.

DINO v1 llega a @@dino.teacher.original.mAP@@ de mAP y @@dino.teacher.original.R1@@ de Rank-1.
El método del paper, con el prior completo, obtiene @@paper.teacher.original.mAP@@ de mAP y
@@paper.teacher.original.R1@@ de Rank-1. La ablación con Sinkhorn-Knopp y KoLeo da
@@paper_dinov2.teacher.original.mAP@@, y el prior suave @@paper_soft.teacher.original.mAP@@.

Todos aprenden: están entre tres y cinco veces por encima del piso. Pero los
valores absolutos son bajos.”

## Slide: Qué vemos

**Tiempo: 13:00–13:50**

“La curva muestra el mAP de validación cada diez épocas. DINO sube de forma
sostenida y todavía no se aplana en la época 100: faltan pasos. El paper
sube más rápido al inicio y se estanca antes.

Esto es consecuencia del régimen: el paper pre-entrena con 4 millones de
imágenes y nosotros con el 0.3 por ciento de eso, en 8 600 pasos. Un ViT
desde cero necesita mucho más. Y el prior hace la tarea del estudiante más
difícil: con tan pocos pasos, eso cuesta precisión.”

## Slide: Baja resolución: la caída de mAP

**Tiempo: 13:50–14:40**

“Aquí está la evidencia a favor del paper. Al reducir las consultas a un
cuarto, DINO cae de @@dino.teacher.original.mAP@@ a @@dino.teacher.low_res.mAP@@ de mAP,
una caída de @@dino.teacher.drop@@ puntos, el @@dino.teacher.drop_rel@@ por ciento de su valor.
El paper cae de @@paper.teacher.original.mAP@@ a @@paper.teacher.low_res.mAP@@, es decir
@@paper.teacher.drop@@ puntos, el @@paper.teacher.drop_rel@@ por ciento.

El estudiante entrenado con imágenes borrosas, reducidas y enmascaradas es
menos sensible a la resolución, que es exactamente lo que el prior promete.
La pregunta es si vale el costo en precisión con consultas limpias.”

## Slide: Ablación: Sinkhorn-Knopp y KoLeo

**Tiempo: 14:40–15:35**

“Con Sinkhorn-Knopp y KoLeo el paper pasa de @@paper.teacher.original.mAP@@ a
@@paper_dinov2.teacher.original.mAP@@ de mAP en test, y de @@paper.val.mAP@@ a
@@paper_dinov2.val.mAP@@ en validación.

Vemos dos razones. Sinkhorn-Knopp obliga a repartir las imágenes por igual
entre los 4 096 prototipos; con 11 mil imágenes, eso son unas tres por
prototipo, y la partición uniforme deja de tener sentido semántico. Y KoLeo
separa los embeddings del batch sin saber si dos de ellos son vistas de la
misma persona, que es justamente lo que queremos juntar.

Estas mejoras están pensadas para cientos de millones de imágenes. Un
resultado negativo también es un resultado de ablación.”

## Slide: ¿Es demasiado agresivo el prior?

**Tiempo: 15:35–16:20**

“El paper eligió una máscara del 60 por ciento con 4 millones de imágenes, y
su Figura 5 muestra que el 30 por ciento rinde casi igual. Con nuestro
estudiante, mucho más débil, ver el 40 por ciento de los patches borrosos y
además reducidos es una tarea muy dura.

Con máscara del 30 por ciento y sin reducción, el método obtiene
@@paper_soft.teacher.original.mAP@@ de mAP y @@paper_soft.teacher.original.R1@@ de Rank-1,
con una caída por baja resolución de @@paper_soft.teacher.drop@@ puntos. Comparado con
@@paper.teacher.original.mAP@@ del prior completo y @@dino.teacher.original.mAP@@ de DINO.
La intensidad del prior es un hiperparámetro que depende del tamaño del
dataset.”

## Slide: Maestro frente a estudiante

**Tiempo: 16:20–16:50**

“El laboratorio pide comparar maestro y estudiante. En DINO, el estudiante da
@@dino.student.original.mAP@@ de mAP frente a @@dino.teacher.original.mAP@@ del maestro. En el
paper, el estudiante da @@paper.student.original.mAP@@ frente a @@paper.teacher.original.mAP@@.
El estudiante del paper nunca vio una imagen limpia durante el entrenamiento y
se evalúa con imágenes limpias; ese desajuste explica que quede por debajo de
su maestro.”

## Slide: ¿A qué atiende el token [CLS]?

**Tiempo: 16:50–17:15**

“Como en la Figura 6 del paper, visualizamos a qué patches atiende el token
CLS en el último bloque, cabeza por cabeza: el softmax del producto entre la
query del CLS y las keys de los 128 patches, reorganizado en la grilla de 16
por 8. [Señalar las columnas.] Las cabezas se reparten regiones de la persona:
torso y ropa, piernas, fondo. Cuando la atención es difusa, el modelo todavía
no ha aprendido qué parte de la imagen identifica a la persona.”

## Slide: Aciertos

**Tiempo: 17:15–17:30**

“Cinco aciertos. En verde, la misma identidad. Son personas con ropa de color
saturado o un accesorio distintivo, como una mochila, visibles desde cualquier
cámara.”

## Slide: Fallos

**Tiempo: 17:30–17:50**

“Cinco fallos. Los vecinos recuperados comparten color de ropa y pose con la
consulta, pero son otras personas: el embedding codifica apariencia global, no
identidad. Ropa oscura y fondos parecidos dan los fallos más sistemáticos, y
las similitudes son altas incluso en los errores, señal de un espacio poco
discriminativo.”

## Slide: Limitaciones y conclusiones

**Tiempo: 17:50–18:00**

“En resumen. Implementamos el método completo desde cero y, con el mismo
código, DINO v1 como base. En test, DINO obtiene @@dino.teacher.original.mAP@@ de mAP y el
paper @@paper.teacher.original.mAP@@, ambos muy por encima del ViT sin entrenar. El prior
reduce la caída por baja resolución, pero cuesta precisión con consultas
limpias. Sinkhorn-Knopp y KoLeo no ayudan con 11 mil imágenes, y un prior más
suave muestra que su intensidad depende del dataset.

Las limitaciones principales son los datos, 11 mil contra 4 millones, la
ausencia de fine-tuning, la validación pequeña y una sola semilla por run.
Gracias.”

---

# Apéndices — solo para preguntas

- Configuración completa: todos los hiperparámetros.
- Código de la implementación: mapa de archivo y símbolo por componente.
- Curvas de validación.
- Aciertos y fallos con consultas de baja resolución.

# Referencia principal

Yang, E., Liu, Y., Zhao, S., Liu, Y., & Liu, S. (2025). Learn from restoration: exploiting task-oriented
knowledge distillation in self-supervised person re-identification. *The Visual Computer*, 41, 6313–6326.
