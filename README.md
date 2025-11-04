# Descripcion del proyecto “Externo”

<img src="logo.png" alt="Diagrama del modelo" width="300"/>


Autor Andres Romo

## 1. Contenido de la carpeta

Esta carpeta contiene un proyecto que muestra el modo de uso del simulador eBallena (para Spiking Neural Networks (SNNs [3]), así como su aplicación en la simulación de un modelo base de Liquid State Machine (LSM [2]) resolviendo el problema de clasificación FSDD (Free Spoken Digits Dataset) [1].

## 2. Objetivo del proyecto

El objetivo del proyecto es mostrar la tecnología que se está utilizando y construyendo a personas pertenecientes a
otros proyectos (distintos de MANIAS). Por esta razón, se intenta presentar ́únicamente aspectos probados y no
funcionalidades en desarrollo. En esta carpeta se pondrán a disposición tutoriales sobre cómo utilizar el simulador
y el modelo LSM baseline.

## 3. Modulos

La carpeta contiene los siguientes submodulos:

- <strong>Baseline</strong>: El objetivo es poder comparar los modelos y tecnología desarrollados con modelos y tecnologías
    probadas y usadas en la literatura. Contiene funciones para contruir una LSM baseline y para simular SNNs
    en el simulador estandar brian2.
- <strong>Data</strong>: Aquí se almacenan los datos raw y datasets. El dataset utilizado en este proyecto es FSDD (free
    spoken digits dataset). Los datos raw de FSDD (audios) deben ser descargado previamente desde [1].
- <strong>eBallena</strong>: Simulador eBallena orientado a eventos. Este simulador esta siendo desarrollado en el proyecto
    MANIAS.
- <strong>Utils</strong> : Funciones útiles de uso recurrente.

## 4. Archivos

En la carpeta principal hay 2 archivos:
- <strong>tutorialEballena.ipynb</strong> el cual contiene un ejemplo minimalista de como usar el simulador eBallena. Se
    recomienda leer el documento documentacioneballena.pdf para complementar.
- <strong>baselinelsm.ipynb</strong> el cual contiene un ejemplo con una liquid state machine resoviendo el problema FSDD.
    
November 3, 2025
## 5. Documentos

En la carpeta documentacion hay 3 documentos.

- <strong>descargaFSDD.pdf</strong> , con instrucciones de como descargar los audios de FSDD.
- <strong>modelolsm.pdf</strong> , con detalles del modelo de Liquid State Machine implementado.
- <strong>documentacion_eballena.ipynb</strong> , con detalles de la implementación del simulador.
    
## 6. Contacto

Cualquier duda o comentario escribir a ai.romo.moena@upm.es

## References

- [1] Zohar Jackson. Free spoken digits dataset. Accessed: 2025-09-29. URL:https://www.kaggle.com/
datasets/joserzapata/free-spoken-digit-dataset-fsdd.

- [2] Wolfgang Maass, Thomas Natschl ̈ager, and Henry Markram. Real-time computing without stable states: A
new framework for neural computation based on perturbations.Neural Computation, 14(11):2531–2560, 11
2002.doi:10.1162/089976602760407955.

- [3] Kai Malcolm and Josue Casco-Rodriguez. A comprehensive review of spiking neural networks: Interpretation,
optimization, efficiency, and best practices.ArXive Preprint, 2023.doi:10.48550/ARXIV.2303.10780.

- [4] Marcel Stimberg, Romain Brette, and Dan FM Goodman. Brian 2, an intuitive and efficient neural simulator.
eLife, 8, Aug 2019. URL:https://brian2.readthedocs.io/en/stable/,doi:10.7554/elife.47314.
