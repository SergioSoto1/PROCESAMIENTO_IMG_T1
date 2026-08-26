# Tarea 1: Compensación de Iluminación y Detección de Personas

**Procesamiento de Imágenes — Universidad Diego Portales (UDP)**

Este repositorio contiene la solución completa para la Tarea 1 de Procesamiento de Imágenes. Se evalúan y comparan diversas técnicas clásicas de mejora de contraste e iluminación y métodos basados en Deep Learning (Zero-DCE) sobre imágenes nocturnas de la base **DARK FACE** (`people/`), midiendo su efectividad cuantitativa a través del detector de personas **YOLOv8**.

---

## 📁 Estructura del Repositorio

- `Tarea_1_Enunciado.ipynb`: Cuaderno Jupyter con todo el código, análisis cuantitativo, gráficos, trípticos y tablas comparativas ejecutadas.
- `utils/funciones.py`: Módulo modular con las funciones de realce clásicas:
  - Corrección Gamma ($\gamma = 0.4$)
  - Estiramiento de contraste lineal por percentiles ($p_5 - p_{95}$)
  - Ecualización global de histograma
  - CLAHE (Ecualización adaptativa con límite de contraste)
  - Función de tríptico (Imagen, Curva de transferencia $T(r)$, Histograma de luminancia $Y$)
- `Zero-DCE/`: Implementación y pesos pre-entrenados del modelo Zero-DCE (`Epoch99.pth`).
- `people/`: Imágenes nocturnas de prueba de la base DARK FACE.
- `resultados/parte_a/`: 90 imágenes resultantes con las anotaciones y detecciones de YOLOv8.

---

## 🚀 Requisitos e Instalación

Para replicar los experimentos localmente:

```bash
pip install torch torchvision opencv-python ultralytics matplotlib pandas numpy ipykernel
```

---

## 📊 Métodos Evaluados

Cada técnica clásica fue evaluada en dos variantes:
1. **BGR (`bgr=True`):** Aplicada independientemente a cada canal $B$, $G$ y $R$.
2. **Luminancia $Y$ (`bgr=False`):** Aplicada únicamente al canal $Y$ en el espacio de color $YCrCb$, preservando la fidelidad cromática original.
3. **Zero-DCE:** Realce adaptativo de iluminación basado en curvas aprendidas por píxel mediante Deep Learning.
