"""
Módulo de funciones auxiliares para procesamiento y realce de imágenes.
Tarea 1 - Procesamiento de Imágenes (UDP)

Funciones implementadas:
  - gamma_transformation      : Corrección gamma (ley de potencias)
  - estiramiento_min_max      : Estiramiento lineal usando mínimo y máximo
  - estiramiento_lo_hi        : Estiramiento lineal por percentiles
  - ecualizacion_histograma   : Ecualización de histograma (implementación propia, sin cv2.equalizeHist)
  - ecualizacion_clahe        : Ecualización adaptativa CLAHE

Alias utilizados por el cuaderno:
  - aplicar_gamma             → gamma_transformation
  - aplicar_estiramiento      → estiramiento_lo_hi  (pct_lo=5, pct_hi=95)
  - aplicar_ecualizacion      → ecualizacion_histograma
  - aplicar_clahe             → ecualizacion_clahe

Helpers de LUT:
  - obtener_lut_gamma
  - obtener_lut_estiramiento
  - obtener_lut_ecualizacion

Visualización:
  - triptico
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers para construir tablas de búsqueda (LUT)
# ---------------------------------------------------------------------------

def obtener_lut_gamma(gamma: float = 0.4) -> np.ndarray:
    """Calcula la LUT para la transformación gamma s = 255·(r/255)^γ."""
    r = np.arange(256, dtype=np.float32)
    s = 255.0 * ((r / 255.0) ** gamma)
    return np.clip(np.round(s), 0, 255).astype(np.uint8)


def obtener_lut_estiramiento(canal: np.ndarray,
                              pct_lo: float = 0.0,
                              pct_hi: float = 100.0) -> np.ndarray:
    """
    Calcula la LUT de estiramiento lineal de contraste.

    El intervalo [p_lo, p_hi] se mapea linealmente a [0, 255].
    Con pct_lo=0 y pct_hi=100 equivale a usar mínimo y máximo.

    Args:
        canal   : Array 2-D (uint8) con los valores de un canal de la imagen.
        pct_lo  : Percentil inferior (default: 0 → mínimo).
        pct_hi  : Percentil superior (default: 100 → máximo).

    Returns:
        LUT de 256 entradas (np.ndarray uint8).
    """
    p_lo = np.percentile(canal, pct_lo)
    p_hi = np.percentile(canal, pct_hi)

    r = np.arange(256, dtype=np.float32)
    if p_hi > p_lo:
        s = (r - p_lo) * (255.0 / (p_hi - p_lo))
    else:
        s = r.copy()
    return np.clip(np.round(s), 0, 255).astype(np.uint8)


def obtener_lut_ecualizacion(canal: np.ndarray) -> np.ndarray:
    """
    Calcula la LUT correspondiente a la ecualización de histograma
    usando la función de distribución acumulada (CDF).

    NO utiliza cv2.equalizeHist: la CDF se calcula con np.histogram.

    Args:
        canal : Array 2-D (uint8) de un canal de la imagen.

    Returns:
        LUT de 256 entradas (np.ndarray uint8).
    """
    # 1. Histograma del canal
    hist, _ = np.histogram(canal.flatten(), bins=256, range=(0, 256))

    # 2. CDF acumulada
    cdf = hist.cumsum()

    # 3. Normalización ignorando los ceros (píxeles sin esa intensidad)
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_norm = (cdf_masked - cdf_masked.min()) * 255.0 / (cdf_masked.max() - cdf_masked.min())
    lut = np.ma.filled(cdf_norm, 0).astype(np.uint8)
    return lut


# ---------------------------------------------------------------------------
# Función interna: aplicar LUT a una imagen respetando la bandera bgr
# ---------------------------------------------------------------------------

def _aplicar_lut_con_bandera(img: np.ndarray,
                              lut: np.ndarray,
                              bgr: bool) -> np.ndarray:
    """
    Aplica una LUT global a la imagen.

    Args:
        img : Imagen BGR uint8.
        lut : Tabla de búsqueda de 256 entradas (uint8).
        bgr : Si True  → aplica la LUT a cada canal B, G, R por separado.
              Si False → aplica la LUT solo al canal Y de YCrCb,
                         preservando la información de color.

    Returns:
        Imagen procesada en formato BGR uint8.
    """
    if bgr:
        return cv2.LUT(img, lut)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# 3.1 Transformación de potencia (gamma)
# ---------------------------------------------------------------------------

def gamma_transformation(img: np.ndarray,
                          gamma: float = 0.4,
                          bgr: bool = True) -> tuple:
    """
    Aplica corrección gamma s = 255·(r/255)^γ.

    Args:
        img   : Imagen BGR uint8.
        gamma : Exponente de la ley de potencias (default: 0.4).
        bgr   : True → aplica a B, G, R por separado.
                False → aplica solo al canal Y (luminancia) de YCrCb.

    Returns:
        (imagen_procesada, lut)
    """
    lut = obtener_lut_gamma(gamma)
    img_proc = _aplicar_lut_con_bandera(img, lut, bgr)
    return img_proc, lut


# Alias usado por el cuaderno
def aplicar_gamma(img: np.ndarray,
                  gamma: float = 0.4,
                  bgr: bool = True) -> np.ndarray:
    """Alias de gamma_transformation; retorna solo la imagen procesada."""
    img_proc, _ = gamma_transformation(img, gamma, bgr)
    return img_proc


# ---------------------------------------------------------------------------
# 3.2a Estiramiento lineal — mínimo y máximo
# ---------------------------------------------------------------------------

def estiramiento_min_max(img: np.ndarray, bgr: bool = True) -> tuple:
    """
    Estiramiento lineal usando los valores mínimo y máximo de la imagen
    como extremos del intervalo de entrada.

    Args:
        img : Imagen BGR uint8.
        bgr : True → percentiles calculados canal a canal (B, G, R).
              False → percentiles sobre el canal Y de YCrCb.

    Returns:
        (imagen_procesada, lut)  — la LUT es la del último canal procesado
        (o del canal Y en el modo bgr=False).
    """
    if bgr:
        canales = list(cv2.split(img))
        canales_proc = []
        lut = None
        for c in canales:
            lut = obtener_lut_estiramiento(c, pct_lo=0.0, pct_hi=100.0)
            canales_proc.append(cv2.LUT(c, lut))
        img_proc = cv2.merge(canales_proc)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lut = obtener_lut_estiramiento(ycrcb[:, :, 0], pct_lo=0.0, pct_hi=100.0)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        img_proc = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_proc, lut


# ---------------------------------------------------------------------------
# 3.2b Estiramiento lineal — percentiles arbitrarios
# ---------------------------------------------------------------------------

def estiramiento_lo_hi(img: np.ndarray,
                        pct_lo: float = 5.0,
                        pct_hi: float = 95.0,
                        bgr: bool = True) -> tuple:
    """
    Estiramiento lineal usando percentiles como extremos del intervalo.

    Args:
        img    : Imagen BGR uint8.
        pct_lo : Percentil inferior (default: 5).
        pct_hi : Percentil superior (default: 95).
        bgr    : True → percentiles calculados canal a canal (B, G, R).
                 False → percentiles sobre el canal Y de YCrCb.

    Returns:
        (imagen_procesada, lut)
    """
    if bgr:
        canales = list(cv2.split(img))
        canales_proc = []
        lut = None
        for c in canales:
            lut = obtener_lut_estiramiento(c, pct_lo, pct_hi)
            canales_proc.append(cv2.LUT(c, lut))
        img_proc = cv2.merge(canales_proc)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lut = obtener_lut_estiramiento(ycrcb[:, :, 0], pct_lo, pct_hi)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        img_proc = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_proc, lut


# Alias usado por el cuaderno
def aplicar_estiramiento(img: np.ndarray,
                          pct_lo: float = 5.0,
                          pct_hi: float = 95.0,
                          bgr: bool = True) -> np.ndarray:
    """Alias de estiramiento_lo_hi; retorna solo la imagen procesada."""
    img_proc, _ = estiramiento_lo_hi(img, pct_lo, pct_hi, bgr)
    return img_proc


# ---------------------------------------------------------------------------
# 3.3 Ecualización de histograma (implementación propia sin cv2.equalizeHist)
# ---------------------------------------------------------------------------

def ecualizacion_histograma(img: np.ndarray, bgr: bool = True) -> tuple:
    """
    Ecualización global de histograma.

    La función de transferencia se construye a partir de la CDF del canal,
    sin usar cv2.equalizeHist.

    Args:
        img : Imagen BGR uint8.
        bgr : True → ecualiza B, G, R por separado.
              False → ecualiza solo el canal Y de YCrCb.

    Returns:
        (imagen_procesada, lut)
    """
    if bgr:
        canales = list(cv2.split(img))
        canales_proc = []
        lut = None
        for c in canales:
            lut = obtener_lut_ecualizacion(c)
            canales_proc.append(cv2.LUT(c, lut))
        img_proc = cv2.merge(canales_proc)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lut = obtener_lut_ecualizacion(ycrcb[:, :, 0])
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        img_proc = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img_proc, lut


# Alias usado por el cuaderno
def aplicar_ecualizacion(img: np.ndarray, bgr: bool = True) -> np.ndarray:
    """Alias de ecualizacion_histograma; retorna solo la imagen procesada."""
    img_proc, _ = ecualizacion_histograma(img, bgr)
    return img_proc


# ---------------------------------------------------------------------------
# 3.4 CLAHE — Ecualización adaptativa con limitación de contraste
# ---------------------------------------------------------------------------

def ecualizacion_clahe(img: np.ndarray,
                        clip_limit: float = 2.0,
                        tile_grid_size: tuple = (8, 8),
                        bgr: bool = True) -> tuple:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Divide la imagen en bloques (tileGridSize) y ecualiza cada bloque
    con un límite de contraste (clipLimit), interpolando bordes.
    Usa cv2.createCLAHE.

    Args:
        img            : Imagen BGR uint8.
        clip_limit     : Umbral de recorte del histograma (default: 2.0).
        tile_grid_size : Tamaño de bloque local (default: (8, 8)).
        bgr            : True → aplica CLAHE a B, G, R por separado.
                         False → aplica CLAHE solo al canal Y de YCrCb.

    Returns:
        (imagen_procesada, lut)  — la LUT siempre es None para CLAHE,
        porque al ser local no existe una única tabla global.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lut = None  # CLAHE no produce una LUT global única

    if bgr:
        b, g, r = cv2.split(img)
        img_proc = cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        img_proc = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return img_proc, lut


# Alias usado por el cuaderno
def aplicar_clahe(img: np.ndarray,
                  clip_limit: float = 2.0,
                  tile_grid_size: tuple = (8, 8),
                  bgr: bool = True) -> np.ndarray:
    """Alias de ecualizacion_clahe; retorna solo la imagen procesada."""
    img_proc, _ = ecualizacion_clahe(img, clip_limit, tile_grid_size, bgr)
    return img_proc


# ---------------------------------------------------------------------------
# Visualización: tríptico (imagen resultante / transformación / histograma Y)
# ---------------------------------------------------------------------------

def triptico(img_original: np.ndarray,
             img_procesada: np.ndarray,
             nombre_metodo: str,
             lut_o_curva: np.ndarray = None,
             curva_label: str = "T(r)",
             figsize: tuple = (15, 4.5)):
    """
    Despliega una figura tipo tríptico:
      1) Imagen resultante procesada.
      2) Función de transformación T(r) vs. línea identidad.
      3) Histograma del canal Y (YCbCr) antes y después del realce.

    Cuando lut_o_curva es None (p. ej. CLAHE o Zero-DCE), el panel
    central muestra la relación empírica de entrada vs. salida como
    nube de puntos.

    Args:
        img_original  : Imagen original en BGR uint8.
        img_procesada : Imagen realzada en BGR uint8.
        nombre_metodo : Etiqueta del método (para títulos).
        lut_o_curva   : LUT de 256 entradas o None.
        curva_label   : Texto de la leyenda de la curva.
        figsize       : Tamaño de la figura (ancho, alto).
    """
    y_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    y_proc = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # ── Panel 1: imagen resultante ──────────────────────────────────────────
    rgb_proc = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB)
    axs[0].imshow(rgb_proc)
    axs[0].set_title(f"Resultado: {nombre_metodo}", fontsize=12, fontweight="bold")
    axs[0].axis("off")

    # ── Panel 2: función de transformación ──────────────────────────────────
    axs[1].plot([0, 255], [0, 255], "k--", alpha=0.5, label="Identidad (s = r)")
    if lut_o_curva is not None:
        r = np.arange(len(lut_o_curva))
        axs[1].plot(r, lut_o_curva, "b-", linewidth=2, label=curva_label)
    else:
        # Relación empírica: muestra 1 de cada 50 píxeles para no saturar
        axs[1].scatter(
            y_orig.flatten()[::50],
            y_proc.flatten()[::50],
            alpha=0.08, s=1, color="steelblue",
            label="Mapeo píxeles (Y)",
        )

    axs[1].set_xlim(0, 255)
    axs[1].set_ylim(0, 255)
    axs[1].set_xlabel("Nivel de entrada r", fontsize=10)
    axs[1].set_ylabel("Nivel de salida s", fontsize=10)
    axs[1].set_title("Función de Transformación T(r)", fontsize=11, fontweight="bold")
    axs[1].legend(loc="lower right", frameon=True)
    axs[1].grid(True, linestyle=":", alpha=0.6)
    axs[1].set_aspect("equal")

    # ── Panel 3: histograma del canal Y ─────────────────────────────────────
    hist_orig, _ = np.histogram(y_orig, bins=256, range=(0, 256))
    hist_proc, _ = np.histogram(y_proc, bins=256, range=(0, 256))

    axs[2].plot(hist_orig, color="gray", linestyle="--", label="Y Original", alpha=0.8)
    axs[2].fill_between(range(256), hist_orig, color="gray", alpha=0.2)
    axs[2].plot(hist_proc, color="crimson", label=f"Y {nombre_metodo}", linewidth=1.5)
    axs[2].fill_between(range(256), hist_proc, color="crimson", alpha=0.2)
    axs[2].set_xlim(0, 255)
    axs[2].set_xlabel("Nivel de Luminancia Y", fontsize=10)
    axs[2].set_ylabel("Frecuencia (píxeles)", fontsize=10)
    axs[2].set_title("Histograma Canal Y (YCbCr)", fontsize=11, fontweight="bold")
    axs[2].legend(loc="upper right", frameon=True)
    axs[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
