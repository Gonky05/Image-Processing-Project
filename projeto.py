import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from modelo import Net  
import torch.nn.functional as F

# CONFIGURAÇÕES
DIM_NORMALIZADA = 400 
CAMINHO_MODELO = "mnist_cnn.pth"  

# ============================================================================
# FUNÇÕES GEOMÉTRICAS 
# ============================================================================
def intersecao_linhas(p1, p2, p3, p4):
    """ Calcula a interseção de duas retas: (p1-p2) e (p3-p4) """
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return p2
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
    tl = pts[np.argmin(s)]; tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]

    if len(pts) == 4:
        br_final = pts[np.argmax(s)]
    elif len(pts) == 5:
        sobras = [p for p in pts if not (np.array_equal(p, tl) or np.array_equal(p, tr) or np.array_equal(p, bl))]
        if len(sobras) == 2:
            p1, p2 = sobras[0], sobras[1]
            if abs(p1[0] - tr[0]) < abs(p2[0] - tr[0]): p_direita, p_baixo = p1, p2
            else: p_direita, p_baixo = p2, p1
            br_final = intersecao_linhas(tr, p_direita, bl, p_baixo)
        else: br_final = pts[np.argmax(s)]
    else: br_final = pts[np.argmax(s)]

    rect[0] = tl; rect[1] = tr; rect[2] = br_final; rect[3] = bl
    return rect

# ============================================================================
# FUNÇÕES PARA DEEP LEARNING (NOVO)
# ============================================================================
def pre_processar_para_cnn(roi_img):
    """ 
    Transforma um recorte (ex: 30x50) num tensor 1x1x28x28 estilo MNIST.
    Mantém o aspect ratio e adiciona padding preto.
    """
    h, w = roi_img.shape
    maior_lado = max(h, w)
    
    # Criar quadrado preto
    quadrado = np.zeros((maior_lado, maior_lado), dtype=np.uint8)
    x_off = (maior_lado - w) // 2
    y_off = (maior_lado - h) // 2
    quadrado[y_off:y_off+h, x_off:x_off+w] = roi_img

    # Adicionar margem de segurança (padding)
    pad = int(maior_lado * 0.2)
    quadrado = cv2.copyMakeBorder(quadrado, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # Redimensionar para 28x28
    img_28 = cv2.resize(quadrado, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalizar
    img_tensor = img_28.astype('float32') / 255.0
    img_tensor = (img_tensor - 0.1307) / 0.3081 # Média/Std padrão do MNIST
    
    # Converter para Tensor PyTorch (Batch, Channel, H, W)
    return torch.from_numpy(img_tensor).unsqueeze(0).unsqueeze(0)

def carregar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A carregar modelo em: {device}")
    model = Net().to(device)
    try:
        # Tenta carregar para GPU ou CPU conforme disponível
        if device.type == 'cpu':
            model.load_state_dict(torch.load(CAMINHO_MODELO, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(CAMINHO_MODELO))
        model.eval()
        return model, device
    except FileNotFoundError:
        print(f"ERRO: Não encontrei '{CAMINHO_MODELO}'. Corre o modelo.py primeiro!")
        exit()

# ============================================================================
# LÓGICA PRINCIPAL
# ============================================================================
def resolver_puzzle_cnn(caminho_img):
    img_bgr = cv2.imread(caminho_img)
    if img_bgr is None: print("Erro: Imagem não encontrada"); return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- PASSO 1-4: SEGMENTAÇÃO E WARP ---
    mask_r1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask_r2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_white = cv2.inRange(hsv, (0, 0, 160), (180, 80, 255))
    
    mask_combined = cv2.bitwise_or(cv2.dilate(mask_red, None), cv2.dilate(mask_white, None))
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    
    largest_contour = max(contours, key=cv2.contourArea)
    rect_src = obter_cantos_extremos(largest_contour)
    M = cv2.getPerspectiveTransform(rect_src, np.array([[0,0], [399,0], [399,399], [0,399]], dtype="float32"))
    warped_color = cv2.warpPerspective(img_rgb, M, (400, 400))
    
    # --- PASSO 5: ISEG ---
    R = warped_color[:,:,0].astype(int)
    G = warped_color[:,:,1].astype(int)
    B = warped_color[:,:,2].astype(int)
    # Ajustei ligeiramente o limiar para ser mais agressivo contra sombras
    is_red = (R > 90) & (R > G + 40) & (R > B + 40)
    is_white = (R > 140) & (G > 140) & (B > 140)
    
    Iseg = np.zeros((400, 400), dtype=np.uint8)
    Iseg[~(is_red | is_white)] = 255 
    
    model, device = carregar_modelo()
    
    # ============================================================================
    # PASSO 6: IDENTIFICAÇÃO COM REGRAS DE LÓGICA
    # ============================================================================
    matriz = []
    res_final = cv2.cvtColor(warped_color, cv2.COLOR_RGB2BGR)
    step = 100 
    kernel_cura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    print("\n--- A LER MATRIZ COM CNN (FILTRADA) ---")

    for i in range(4):
        linha = []
        for j in range(4):
            # Recorte da célula
            cel_bin = Iseg[i*step:(i+1)*step, j*step:(j+1)*step].copy()
            
            # Limpeza de bordas agressiva (15px) para remover linhas da grelha
            cel_bin[0:15, :] = 0; cel_bin[85:100, :] = 0
            cel_bin[:, 0:15] = 0; cel_bin[:, 85:100] = 0
            
            cel_bin = cv2.morphologyEx(cel_bin, cv2.MORPH_CLOSE, kernel_cura)

            cnts_n, _ = cv2.findContours(cel_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            digitos_detectados = [] # Lista de tuplos: (x, valor_predito, area, confiança)
            
            area_celula = step * step # 100x100 = 10000
            
            for c in cnts_n:
                area = cv2.contourArea(c)
                # REGRA 1: Filtro de Ruído Relativo
                # Só aceita se o contorno for maior que 2.5% da célula (aprox 250px)
                # Isto elimina os "pontinhos" que a CNN acha que são 1s
                if area > (0.025 * area_celula): 
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Filtro de Aspect Ratio (Opcional): Elimina linhas muito finas
                    ratio = w / float(h)
                    if 0.15 < ratio < 1.5: # Dígitos normais estão neste intervalo
                        roi = cel_bin[y:y+h, x:x+w]
                        tensor_input = pre_processar_para_cnn(roi).to(device)
                        
                        with torch.no_grad():
                            output = model(tensor_input)
                            # Calcular probabilidades (Softmax)
                            probs = F.softmax(output, dim=1)
                            confianca, pred = torch.max(probs, 1)
                            
                        val = pred.item()
                        conf = confianca.item()
                        
                        # REGRA 2: Confiança Mínima
                        # Se a CNN tiver menos de 60% de certeza, ignoramos
                        if conf > 0.6:
                            digitos_detectados.append({
                                'x': x, 
                                'val': val, 
                                'area': area,
                                'box': (x, y, w, h)
                            })

            # Ordenar pela posição X (esquerda para direita)
            digitos_detectados.sort(key=lambda k: k['x'])

            val_final = -1
            
            if len(digitos_detectados) == 0:
                val_final = -1 # Vazio
            
            elif len(digitos_detectados) == 1:
                val_final = digitos_detectados[0]['val']
            
            else:
                # Temos 2 ou mais dígitos. Vamos construir o número.
                # Se tivermos mais de 2, algo está errado (ruído). Ficamos com os 2 maiores.
                if len(digitos_detectados) > 2:
                    digitos_detectados.sort(key=lambda k: k['area'], reverse=True)
                    digitos_detectados = digitos_detectados[:2] # Pega nos 2 maiores
                    digitos_detectados.sort(key=lambda k: k['x']) # Reordena por X

                # Tenta combinar
                num_str = str(digitos_detectados[0]['val']) + str(digitos_detectados[1]['val'])
                val_final = int(num_str)

                # REGRA 3: O limite é 15
                if val_final > 15:
                    # Se deu > 15 (ex: "60" ou "47"), é impossível neste puzzle.
                    # Lógica: O dígito verdadeiro é provavelmente o que tem maior área (mais pixels)
                    # O outro era ruído que passou pelos filtros.
                    maior_digito = max(digitos_detectados, key=lambda k: k['area'])
                    val_final = maior_digito['val']
                    
                    # Nota: Isto corrige o "60" virar "6" e o "168" virar "8" (ou o maior deles)

            linha.append(val_final)
            
            # --- Visualização ---
            if val_final != -1:
                cor = (0, 255, 0)
                txt = str(val_final)
                # Desenha caixa apenas nos dígitos usados
                for d in digitos_detectados:
                    if str(d['val']) in txt: # Se contribuiu para o final
                        (rx, ry, rw, rh) = d['box']
                        cv2.rectangle(res_final, (j*step+rx, i*step+ry), (j*step+rx+rw, i*step+ry+rh), (0,255,255), 1)
            else:
                cor = (255, 0, 0)
                txt = "-1"
            
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(res_final, txt, (j*step + (100-tw)//2, i*step + (100+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)

        matriz.append(linha)

    # OUTPUTS
    print("\n" + "="*30)
    print(" MATRIZ DETETADA (CORRIGIDA)")
    print("="*30)
    for l in matriz: print(l)
    print("="*30 + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(Iseg, cmap='gray'); axes[0].set_title("Segmentação")
    axes[1].imshow(cv2.cvtColor(res_final, cv2.COLOR_BGR2RGB)); axes[1].set_title("Resultado Final")
    plt.show()

# Executar
if __name__ == "__main__":
    # Substitui pelo nome da tua imagem
    resolver_puzzle_cnn("imagem_quadrado2.jpg")