import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# CONFIGURAÇÕES
DIM_NORMALIZADA = 400 
PASTA_REFERENCIAS = "referencias"

def intersecao_linhas(p1, p2, p3, p4):
    """ Calcula a interseção de duas retas: (p1-p2) e (p3-p4) """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: # Linhas paralelas
        return p2
        
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return [px, py]

def obter_cantos_extremos(pts_contorno):
    
    peri = cv2.arcLength(pts_contorno, True)
    approx = cv2.approxPolyDP(pts_contorno, 0.02 * peri, True)
    pts = approx.reshape(-1, 2)
    
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if len(pts) == 4:
        br_final = pts[np.argmax(s)]

    
    elif len(pts) == 5:
        
        sobras = []
        for p in pts:
            if not (np.array_equal(p, tl) or np.array_equal(p, tr) or np.array_equal(p, bl)):
                sobras.append(p)
        
        
        if len(sobras) == 2:
            p1, p2 = sobras[0], sobras[1]

            
            
            if abs(p1[0] - tr[0]) < abs(p2[0] - tr[0]):
                p_direita = p1
                p_baixo = p2
            else:
                p_direita = p2
                p_baixo = p1
            
            
            br_final = intersecao_linhas(tr, p_direita, bl, p_baixo)
        else:
            
            br_final = pts[np.argmax(s)]
            
    
    else:
        
        br_final = pts[np.argmax(s)]

    rect[0] = tl
    rect[1] = tr
    rect[2] = br_final
    rect[3] = bl
    
    return rect

def resolver_puzzle_final(caminho_img, templates):
    img_bgr = cv2.imread(caminho_img)
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ============================================================================
    # PASSO 1 & 2: SEGMENTAÇÃO HSV - tabuleiro
    # ============================================================================
    mask_r1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask_r2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_white = cv2.inRange(hsv, (0, 0, 160), (180, 80, 255))

    
    mask_red = cv2.medianBlur(mask_red, 5)
    mask_white = cv2.medianBlur(mask_white, 5)

    # ============================================================================
    # PASSO 3: ISOLAMENTO DO TABULEIRO (Interceção)
    # ============================================================================ 
    mask_r_d = cv2.dilate(mask_red, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    mask_w_d = cv2.dilate(mask_white, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    # Encontrar onde as peças vermelhas e brancas se intersetam
    inter = cv2.bitwise_and(mask_r_d, mask_w_d)
    
    num_r, labels_r = cv2.connectedComponents(mask_r_d)
    num_w, labels_w = cv2.connectedComponents(mask_w_d)

    # Filtrar apenas componentes que fazem parte do aglomerado (tabuleiro)
    ids_r = np.unique(labels_r[inter > 0])
    ids_w = np.unique(labels_w[inter > 0])
    ids_r = ids_r[ids_r != 0]; ids_w = ids_w[ids_w != 0]

    mask_r_touch = np.zeros_like(labels_r, dtype=np.uint8)
    for rid in ids_r: mask_r_touch |= (labels_r == rid).astype(np.uint8)
    
    mask_w_touch = np.zeros_like(labels_w, dtype=np.uint8)
    for wid in ids_w: mask_w_touch |= (labels_w == wid).astype(np.uint8)

    # União das peças que se tocam
    mask_tiles_touch = cv2.bitwise_or(mask_r_touch, mask_w_touch)
    mask_tiles_touch = (mask_tiles_touch * 255).astype(np.uint8)

    # CRIAR MÁSCARA SÓLIDA DO Tabuleiro
    contours, _ = cv2.findContours(mask_tiles_touch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    
    largest_contour = max(contours, key=cv2.contourArea)
    mask_pecas = np.zeros_like(mask_tiles_touch)
    
    # Desenha o contorno preenchido para ignorar o fundo
    cv2.drawContours(mask_pecas, [largest_contour], -1, 255, thickness=-1)

    # ============================================================================
    # PASSO 4: WARP Tabuleiro
    # ============================================================================
    rect_src = obter_cantos_extremos(largest_contour)
    M = cv2.getPerspectiveTransform(rect_src, np.array([[0,0], [399,0], [399,399], [0,399]], dtype="float32"))
    warped_color = cv2.warpPerspective(img_rgb, M, (400, 400))
    warped_mask_tabuleiro = cv2.warpPerspective(mask_pecas, M, (400, 400))

    # ============================================================================
    # PASSO 5: ISEG - Segmentação dos números
    # ============================================================================
    R = warped_color[:,:,0].astype(int)
    G = warped_color[:,:,1].astype(int)
    B = warped_color[:,:,2].astype(int)

    
    # Se tem muito vermelho e se o R é maior que G e B
    is_red = (R > 80) & (R > G + 40) & (R > B + 40)

    # Regra para Branco: Valores altos em todos os canais
    is_white = (R > 150) & (G > 150) & (B > 150)

  
    Iseg = np.zeros((400, 400), dtype=np.uint8)

    
    Iseg[~(is_red | is_white)] = 255

    
    warped_mask_tabuleiro = warped_mask_tabuleiro.astype(np.uint8)
    Iseg = cv2.bitwise_and(Iseg, warped_mask_tabuleiro)

    # Dilatação para unir partes falhadas dos números
    Iseg = cv2.dilate(Iseg, np.ones((3,3), np.uint8), iterations=1)

    # ============================================================================
    # PASSO 6: Identificar números
    # ============================================================================
    matriz = []
    res_final = warped_color.copy()
    step = 100 

    # Kernel para fechar pequenos buracos nos números
    kernel_cura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    for i in range(4):
        linha = []
        for j in range(4):
            # --- 1. PREPARAÇÃO DA CÉLULA ---
            cel_bin = Iseg[i*step:(i+1)*step, j*step:(j+1)*step].copy()
            
            
            h_cel, w_cel = cel_bin.shape
            margem = 12
            cel_bin[0:margem, :] = 0; cel_bin[h_cel-margem:h_cel, :] = 0
            cel_bin[:, 0:margem] = 0; cel_bin[:, w_cel-margem:w_cel] = 0

           
            cel_bin = cv2.morphologyEx(cel_bin, cv2.MORPH_CLOSE, kernel_cura)

            # --- 2. LÓGICA DE DETEÇÃO ---
            num_identificado = -1
            melhor_score = 0.40  
            
            cnts_n, _ = cv2.findContours(cel_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtra ruído
            cnts_validos = [c for c in cnts_n if cv2.contourArea(c) > 20]

            if cnts_validos:
                
                min_x, min_y = 1000, 1000
                max_x, max_y = 0, 0

                for c in cnts_validos:
                    x, y, w, h = cv2.boundingRect(c)
                    if x < min_x: min_x = x
                    if y < min_y: min_y = y
                    if x + w > max_x: max_x = x + w
                    if y + h > max_y: max_y = y + h
                
                # Padding (Margem de segurança para o matchTemplate não falhar nas bordas)
                pad = 4
                nx = max(0, min_x - pad)
                ny = max(0, min_y - pad)
                nw = min(w_cel - nx, (max_x - min_x) + 2*pad)
                nh = min(h_cel - ny, (max_y - min_y) + 2*pad)

                recorte_num = cel_bin[ny:ny+nh, nx:nx+nw]

                if nw > 5 and nh > 10: 
                    for val, temp in templates.items():
                        temp_res = cv2.resize(temp, (nw, nh))
                        res = cv2.matchTemplate(recorte_num, temp_res, cv2.TM_CCOEFF_NORMED)
                        _, score, _, _ = cv2.minMaxLoc(res)
                        
                        
                        ratio = nw / float(nh)
                        if ratio > 0.8 and val >= 10: score += 0.12 
                        elif ratio < 0.6 and val < 10: score += 0.10 

                        if score > melhor_score:
                            melhor_score = score
                            num_identificado = val

            linha.append(num_identificado)
            
            
            if num_identificado != -1:
                
                cor = (0, 255, 0)
                txt = str(num_identificado)
            else:
                
                cor = (255, 0, 0) 
                txt = "-1"

            
            (text_w, text_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            pos_x = j * step + (step - text_w) // 2
            pos_y = i * step + (step + text_h) // 2
            
            cv2.putText(res_final, txt, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)

        matriz.append(linha)

    # ============================================================================
    # OUTPUTS
    # ============================================================================
    
    # 1. ESCREVER MATRIZ NA CONSOLA
    print("\n" + "="*30)
    print(" MATRIZ DO TABULEIRO DETETADA")
    print("="*30)
    for l in matriz:
        print(l)
    print("="*30 + "\n")

    # 2. MOSTRAR IMAGENS
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].imshow(mask_red, cmap='gray'); axes[0,0].set_title("Segmentação dos Vermelhos")
    axes[0,1].imshow(mask_white, cmap='gray'); axes[0,1].set_title("Segmentação dos Brancos")
    axes[0,2].imshow(mask_pecas, cmap='gray'); axes[0,2].set_title("Tabuleiro Segmentado")
    axes[1,0].imshow(warped_color); axes[1,0].set_title("Vista Frontal - Tabuleiro")
    axes[1,1].imshow(Iseg, cmap='gray'); axes[1,1].set_title("Números segmentados")
    axes[1,2].imshow(res_final); axes[1,2].set_title("Resultado Final")
    plt.show()



# Carregamento de templates
templates = {}
for i in range(1, 16):
    p = os.path.join(PASTA_REFERENCIAS, f"mask{i}.png")
    if os.path.exists(p):
        templates[i] = cv2.threshold(cv2.imread(p, 0), 127, 255, cv2.THRESH_BINARY)[1]

resolver_puzzle_final("imagem_quadrado.jpg", templates)