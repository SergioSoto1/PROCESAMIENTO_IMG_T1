"""
Módulo de funciones auxiliares para procesamiento y realce de imágenes.
Tarea 1 - Procesamiento de Imágenes (UDP)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def obtener_lut_gamma(gamma: float = 0.4) -> np.ndarray:
    """Calcula la tabla de búsqueda (LUT) para la transformación gamma."""
    r = np.arange(256, dtype=np.float32)
    s = 255.0 * ((r / 255.0) ** gamma)
    return np.clip(np.round(s), 0, 255).astype(np.uint8)


def aplicar_gamma(img: np.ndarray, gamma: float = 0.4, bgr: bool = True) -> np.ndarray:
    """
    Aplica corrección gamma (ley de potencias) a una imagen.
    
    Args:
        img: Imagen en formato BGR (uint8).
        gamma: Exponente gamma (default: 0.4).
        bgr: Si True, aplica a cada canal BGR por separado.
             Si False, aplica únicamente al canal Y (luminancia) en espacio YCrCb.
             
    Returns:
        Imagen realzada en formato BGR (uint8).
    """
    lut = obtener_lut_gamma(gamma)
    
    if bgr:
        return cv2.LUT(img, lut)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def obtener_lut_estiramiento(canal: np.ndarray, pct_lo: float = 5.0, pct_hi: float = 95.0) -> np.ndarray:
    """Calcula la LUT de estiramiento de contraste lineal por percentiles para un canal dado."""
    p_lo = np.percentile(canal, pct_lo)
    p_hi = np.percentile(canal, pct_hi)
    
    r = np.arange(256, dtype=np.float32)
    if p_hi > p_lo:
        s = (r - p_lo) * (255.0 / (p_hi - p_lo))
    else:
        s = r
    return np.clip(np.round(s), 0, 255).astype(np.uint8)


def aplicar_estiramiento(img: np.ndarray, pct_lo: float = 5.0, pct_hi: float = 95.0, bgr: bool = True) -> np.ndarray:
    """
    Aplica estiramiento de contraste por percentiles (linear percentile stretching).
    
    Args:
        img: Imagen en formato BGR (uint8).
        pct_lo: Percentil inferior (default: 5).
        pct_hi: Percentil superior (default: 95).
        bgr: Si True, calcula percentiles y estira cada canal B, G, R de forma independiente.
             Si False, opera exclusivamente sobre el canal Y de YCrCb.
             
    Returns:
        Imagen realzada en formato BGR (uint8).
    """
    if bgr:
        canales = cv2.split(img)
        canales_proc = []
        for c in canales:
            lut = obtener_lut_estiramiento(c, pct_lo, pct_hi)
            canales_proc.append(cv2.LUT(c, lut))
        return cv2.merge(canales_proc)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lut = obtener_lut_estiramiento(ycrcb[:, :, 0], pct_lo, pct_hi)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def obtener_lut_ecualizacion(canal: np.ndarray) -> np.ndarray:
    """Calcula la LUT correspondiente a la función de distribución acumulada (CDF) del canal."""
    hist, _ = np.histogram(canal.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    lut = np.ma.filled(cdf_m, 0).astype(np.uint8)
    return lut


def aplicar_ecualizacion(img: np.ndarray, bgr: bool = True) -> np.ndarray:
    """
    Aplica ecualización de histograma global (Histogram Equalization).
    
    Args:
        img: Imagen en formato BGR (uint8).
        bgr: Si True, ecualiza cada canal B, G, R por separado.
             Si False, ecualiza únicamente el canal Y de YCrCb.
             
    Returns:
        Imagen realzada en formato BGR (uint8).
    """
    if bgr:
        b, g, r = cv2.split(img)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge([b_eq, g_eq, r_eq])
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def aplicar_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8), bgr: bool = True) -> np.ndarray:
    """
    Aplica ecualización adaptativa de histograma con límite de contraste (CLAHE).
    
    Args:
        img: Imagen en formato BGR (uint8).
        clip_limit: Umbral de corte para el límite de contraste (default: 2.0).
        tile_grid_size: Tamaño de la rejilla de bloques locales (default: (8, 8)).
        bgr: Si True, aplica CLAHE a cada canal BGR por separado.
             Si False, aplica CLAHE solo al canal Y de YCrCb.
             
    Returns:
        Imagen realzada en formato BGR (uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if bgr:
        b, g, r = cv2.split(img)
        b_clahe = clahe.apply(b)
        g_clahe = clahe.apply(g)
        r_clahe = clahe.apply(r)
        return cv2.merge([b_clahe, g_clahe, r_clahe])
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def triptico(img_original: np.ndarray,
             img_procesada: np.ndarray,
             nombre_metodo: str,
             lut_o_curva: np.ndarray = None,
             curva_label: str = "T(r)",
             figsize: tuple = (15, 4.5)):
    """
    Despliega la figura tipo tríptico:
    1) Imagen resultante procesada.
    2) Función de transformación T(r) vs Identidad.
    3) Histograma del canal Y (YCbCr) antes y después.
    """
    y_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    y_proc = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Imagen Resultante
    rgb_proc = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB)
    axs[0].imshow(rgb_proc)
    axs[0].set_title(f"Resultado: {nombre_metodo}", fontsize=12, fontweight='bold')
    axs[0].axis('off')
    
    # 2. Función de Transformación
    axs[1].plot([0, 255], [0, 255], 'k--', alpha=0.5, label='Identidad (s = r)')
    if lut_o_curva is not None:
        r = np.arange(len(lut_o_curva))
        axs[1].plot(r, lut_o_curva, 'b-', linewidth=2, label=curva_label)
    else:
        # Aproximación empírica de entrada vs salida promedio sobre Y
        axs[1].scatter(y_orig.flatten()[::50], y_proc.flatten()[::50], alpha=0.1, s=1, color='blue', label='Mapeo píxeles')
        
    axs[1].set_xlim(0, 255)
    axs[1].set_ylim(0, 255)
    axs[1].set_xlabel("Nivel de entrada r", fontsize=10)
    axs[1].set_ylabel("Nivel de salida s", fontsize=10)
    axs[1].set_title("Función de Transformación T(r)", fontsize=11, fontweight='bold')
    axs[1].legend(loc='lower right', frameon=True)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].set_aspect('equal')
    
    # 3. Histograma Canal Y
    hist_orig, _ = np.histogram(y_orig, bins=256, range=(0, 256))
    hist_proc, _ = np.histogram(y_proc, bins=256, range=(0, 256))
    
    axs[2].plot(hist_orig, color='gray', linestyle='--', label='Y Original', alpha=0.8)
    axs[2].fill_between(range(256), hist_orig, color='gray', alpha=0.2)
    axs[2].plot(hist_proc, color='crimson', label=f'Y {nombre_metodo}', linewidth=1.5)
    axs[2].fill_between(range(256), hist_proc, color='crimson', alpha=0.2)
    axs[2].set_xlim(0, 255)
    axs[2].set_xlabel("Nivel de Luminancia Y", fontsize=10)
    axs[2].set_ylabel("Frecuencia (píxeles)", fontsize=10)
    axs[2].set_title("Histograma Canal Y (YCbCr)", fontsize=11, fontweight='bold')
    axs[2].legend(loc='upper right', frameon=True)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
