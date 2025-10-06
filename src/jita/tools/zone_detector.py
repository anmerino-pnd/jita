import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

class HierarchicalFlyerDetector:
    """
    Detector jerárquico optimizado para folletos de supermercado.
    Detecta columnas principales y las subdivide horizontalmente con padding inteligente.
    """
    
    def __init__(self, 
                 primary_divisions='auto',
                 subdivisions=3,
                 padding_percent=5,
                 preprocessing='balanced'):
        """
        Args:
            primary_divisions: 'auto' o número fijo (2, 3, etc.)
            subdivisions: Cuántas zonas crear dentro de cada columna
            padding_percent: Porcentaje de padding arriba/abajo (0-20 recomendado)
            preprocessing: 'gentle', 'balanced', 'aggressive'
        """
        self.primary_divisions = primary_divisions
        self.subdivisions = subdivisions
        self.padding_percent = padding_percent
        self.preprocessing_preset = preprocessing
    
    def detect_zones(self, image_path, output_folder, debug=False):
        """
        Detecta y recorta zonas con padding inteligente.
        
        Returns:
            Lista de diccionarios con info de cada zona
        """
        os.makedirs(output_folder, exist_ok=True)
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Preprocesamiento
        gray = self._preprocess_image(img)
        
        if debug:
            cv2.imwrite(os.path.join(output_folder, "0_preprocessed.jpg"), gray)
            print(f"📸 Imagen preprocesada: {w}x{h}px")
        
        # NIVEL 1: Detectar divisiones principales (columnas)
        print(f"\n🔍 Detectando divisiones principales...")
        primary_zones = self._detect_primary_divisions(img, gray, debug, output_folder)
        print(f"   ✓ {len(primary_zones)} columnas detectadas")
        
        # NIVEL 2: Subdividir cada columna con padding
        print(f"\n🔍 Subdividiendo en {self.subdivisions} zonas c/u (padding: {self.padding_percent}%)...")
        all_zones = []
        zone_counter = 1
        
        for col_idx, (px, py, pw, ph) in enumerate(primary_zones):
            print(f"   → Columna {col_idx + 1}: procesando...")
            
            # Extraer región de la columna
            column_img = img[py:py+ph, px:px+pw]
            column_gray = gray[py:py+ph, px:px+pw]
            
            # Subdividir
            sub_zones = self._subdivide_zone(
                column_img, column_gray, 
                self.subdivisions, debug, output_folder, col_idx
            )
            
            # Aplicar padding y recortar
            for (sx, sy, sw, sh) in sub_zones:
                # Calcular padding basado en altura de la zona
                padding_top = int(sh * self.padding_percent / 100)
                padding_bottom = int(sh * self.padding_percent / 100)
                
                # Coordenadas globales con padding
                global_x = px + sx
                global_y = max(0, py + sy - padding_top)  # No salir de la imagen
                global_h = min(h - global_y, sh + padding_top + padding_bottom)
                
                # Recortar con padding
                crop = img[global_y:global_y+global_h, global_x:global_x+sw]
                
                # Guardar
                out_path = os.path.join(output_folder, f"zone_{zone_counter:02d}.jpg")
                cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                all_zones.append({
                    'id': zone_counter,
                    'column': col_idx + 1,
                    'x': global_x,
                    'y': global_y,
                    'width': sw,
                    'height': global_h,
                    'path': out_path,
                    'padding_applied': padding_top + padding_bottom
                })
                zone_counter += 1
            
            print(f"      ✓ {len(sub_zones)} zonas creadas")
        
        print(f"\n✅ Total: {len(all_zones)} zonas listas")
        
        # Visualización final
        if debug:
            self._create_debug_visualization(img, all_zones, output_folder)
        
        return all_zones
    
    def _preprocess_image(self, img):
        """Preprocesamiento optimizado."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Aplicar preset
        if self.preprocessing_preset == 'aggressive':
            pil_img = ImageEnhance.Contrast(pil_img).enhance(2.5)
            pil_img = ImageEnhance.Color(pil_img).enhance(1.8)
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
            pil_img = ImageEnhance.Brightness(pil_img).enhance(1.2)
            
        elif self.preprocessing_preset == 'balanced':
            pil_img = ImageEnhance.Contrast(pil_img).enhance(1.8)
            pil_img = ImageEnhance.Color(pil_img).enhance(1.4)
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
            
        elif self.preprocessing_preset == 'gentle':
            pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
            pil_img = ImageEnhance.Color(pil_img).enhance(1.2)
        
        # Convertir a gris con CLAHE
        enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return gray
    
    def _detect_primary_divisions(self, img, gray, debug, output_folder):
        """Detecta columnas principales usando proyección vertical."""
        h, w = img.shape[:2]
        
        # Proyección vertical (suma de píxeles oscuros por columna)
        v_projection = np.sum(gray < 200, axis=0)
        
        # Suavizar proyección para eliminar ruido
        v_projection_smooth = np.convolve(v_projection, np.ones(w//50)//(w//50), mode='same')
        
        # Umbral adaptativo
        v_threshold = np.max(v_projection_smooth) * 0.12
        
        if debug:
            self._save_projection_plot(
                v_projection_smooth, v_threshold, 
                'vertical', output_folder,
                "Detección de Columnas (Proyección Vertical)"
            )
        
        # Encontrar gaps (separadores entre columnas)
        gaps = self._find_gaps(v_projection_smooth, v_threshold, min_gap_size=w//30)
        
        # Crear divisiones
        if self.primary_divisions == 'auto':
            x_splits = [0] + gaps + [w]
            
            # Si no hay gaps, dividir en 2
            if len(x_splits) == 2:
                x_splits = [0, w//2, w]
                print("   ⚠️  No se detectaron separadores, dividiendo en 2 columnas")
            
            # Limitar a máximo 4 columnas (heurística)
            if len(x_splits) > 5:
                # Tomar solo los gaps más pronunciados
                gap_strengths = []
                for gap in gaps:
                    strength = abs(v_projection_smooth[gap] - v_threshold)
                    gap_strengths.append((gap, strength))
                gap_strengths.sort(key=lambda x: x[1], reverse=True)
                top_gaps = sorted([g[0] for g in gap_strengths[:3]])
                x_splits = [0] + top_gaps + [w]
        else:
            # División manual en N partes iguales
            n = self.primary_divisions
            x_splits = [int(w * i / n) for i in range(n + 1)]
        
        # Generar zonas (columnas completas de arriba a abajo)
        primary_zones = []
        for i in range(len(x_splits) - 1):
            x1, x2 = x_splits[i], x_splits[i + 1]
            primary_zones.append((x1, 0, x2 - x1, h))
        
        return primary_zones
    
    def _subdivide_zone(self, region_img, region_gray, n_subdivisions, 
                       debug, output_folder, col_idx):
        """Subdivide una columna horizontalmente."""
        h, w = region_img.shape[:2]
        
        # Proyección horizontal
        h_projection = np.sum(region_gray < 200, axis=1)
        
        # Suavizar proyección
        window = max(5, h//100)
        h_projection_smooth = np.convolve(h_projection, np.ones(window)/window, mode='same')
        
        # Umbral más sensible para subdivisiones
        h_threshold = np.max(h_projection_smooth) * 0.08
        
        if debug:
            self._save_projection_plot(
                h_projection_smooth, h_threshold,
                f'horizontal_col{col_idx+1}', output_folder,
                f"Subdivisiones Columna {col_idx+1} (Proyección Horizontal)",
                orientation='horizontal'
            )
        
        # Encontrar gaps horizontales
        h_gaps = self._find_gaps(h_projection_smooth, h_threshold, min_gap_size=h//25)
        
        # Crear subdivisiones
        y_splits = [0] + h_gaps + [h]
        
        # Ajustar al número deseado
        if len(y_splits) - 1 > n_subdivisions:
            # Tomar splits más espaciados
            indices = np.linspace(0, len(y_splits) - 1, n_subdivisions + 1, dtype=int)
            y_splits = [y_splits[i] for i in indices]
        elif len(y_splits) - 1 < n_subdivisions:
            # División uniforme si hay muy pocos gaps
            y_splits = [int(h * i / n_subdivisions) for i in range(n_subdivisions + 1)]
        
        # Generar sub-zonas
        sub_zones = []
        for i in range(len(y_splits) - 1):
            y1, y2 = y_splits[i], y_splits[i + 1]
            
            # Filtrar zonas muy pequeñas (menos del 5% de la altura)
            if (y2 - y1) > h * 0.05:
                sub_zones.append((0, y1, w, y2 - y1))
        
        return sub_zones
    
    def _find_gaps(self, projection, threshold, min_gap_size):
        """Encuentra espacios vacíos (separadores)."""
        below_threshold = projection < threshold
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, is_gap in enumerate(below_threshold):
            if is_gap and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_gap and in_gap:
                gap_length = i - gap_start
                if gap_length >= min_gap_size:
                    # Tomar el punto medio del gap
                    gaps.append((gap_start + i) // 2)
                in_gap = False
        
        return gaps
    
    def _save_projection_plot(self, projection, threshold, name, output_folder, 
                             title, orientation='vertical'):
        """Guarda gráfica de proyección."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 5) if orientation == 'vertical' else (6, 10))
            
            if orientation == 'vertical':
                plt.plot(projection, linewidth=2, color='#2E86AB')
                plt.axhline(y=threshold, color='#A23B72', linestyle='--', 
                           linewidth=2, label=f'Umbral ({threshold:.0f})')
                plt.xlabel('Posición X (píxeles)', fontsize=11)
                plt.ylabel('Intensidad de contenido', fontsize=11)
            else:
                plt.plot(projection, range(len(projection)), linewidth=2, color='#2E86AB')
                plt.axvline(x=threshold, color='#A23B72', linestyle='--', 
                           linewidth=2, label=f'Umbral ({threshold:.0f})')
                plt.ylabel('Posición Y (píxeles)', fontsize=11)
                plt.xlabel('Intensidad de contenido', fontsize=11)
                plt.gca().invert_yaxis()
            
            plt.title(title, fontsize=13, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3, linestyle=':')
            plt.tight_layout()
            
            out_path = os.path.join(output_folder, f"1_projection_{name}.png")
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            pass  # Matplotlib no disponible
    
    def _create_debug_visualization(self, img, zones, output_folder):
        """Crea visualización final con todas las zonas marcadas."""
        debug_img = img.copy()
        
        # Colores por columna
        colors = [
            (255, 50, 50),   # Rojo
            (50, 255, 50),   # Verde
            (50, 50, 255),   # Azul
            (255, 255, 50),  # Amarillo
            (255, 50, 255),  # Magenta
        ]
        
        for zone in zones:
            color = colors[(zone['column'] - 1) % len(colors)]
            
            # Dibujar rectángulo
            x, y = zone['x'], zone['y']
            w, h = zone['width'], zone['height']
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 4)
            
            # Etiqueta
            label = f"Z{zone['id']}"
            cv2.putText(debug_img, label, (x + 10, y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Info de padding (esquina superior derecha de cada zona)
            padding_info = f"+{zone['padding_applied']}px"
            cv2.putText(debug_img, padding_info, (x + w - 100, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_path = os.path.join(output_folder, "2_final_zones.jpg")
        cv2.imwrite(out_path, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\n📊 Visualización guardada: {out_path}")
