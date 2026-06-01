import cv2
import numpy as np
import os
import glob
import tkinter as tk
import math
import time
from itertools import combinations
from PIL import Image, ImageTk

# --- Configurações Iniciais do Estado (Seus Parâmetros Calibrados) ---
ESTADO = {
    'idx': 0,
    # 1. Filtros de Cor Laranja (HSV)
    'h1_min': 0, 'h1_max': 19,
    'h2_min': 175, 'h2_max': 179,
    's_min': 140, 's_max': 255,  
    'v_min': 100, 'v_max': 255,  
    # 2. Morfologia Matemática
    'k_abertura': 3, 'iter_abertura': 1,
    # 3. Heurísticas de Base (Blocos)
    'area_minima': 5,
    'fator_proporcao': 0.012,    
    # 4 e 5. Regras de Densidade e Espaciais (K-NN)
    'limiar_laranja_min': 0.15,  
    'limiar_laranja_max': 1.00,  
    'limiar_total': 0.24,        
    'k_vizinhos': 7,             
    'fator_raio': 2.40,
    
    # --- Limiares de IoU para Separação de Dataset ---
    'iou_positivo': 0.49,
    'iou_negativo': 0.20
}

caminhos_imagens = []
img_original_atual = None
tk_image = None

def carregar_dataset(pasta):
    global caminhos_imagens
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        caminhos_imagens.extend(glob.glob(os.path.join(pasta, ext)))
    caminhos_imagens.sort()
    return len(caminhos_imagens)

def ler_gabarito_yolo(img_path, img_w, img_h):
    base_nome = os.path.splitext(os.path.basename(img_path))[0]
    dir_img = os.path.dirname(img_path)
    dir_label = dir_img.replace('images', 'labels')
    txt_path = os.path.join(dir_label, base_nome + '.txt')
    
    caixas_gabarito = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for linha in f:
                partes = linha.strip().split()
                if len(partes) >= 5:
                    cl, x_c, y_c, w, h = map(float, partes[:5])
                    x_px = int((x_c - w/2) * img_w)
                    y_px = int((y_c - h/2) * img_h)
                    w_px = int(w * img_w)
                    h_px = int(h * img_h)
                    caixas_gabarito.append((x_px, y_px, w_px, h_px))
    return caixas_gabarito

def calcular_iou(boxA, boxB):
    xA_min, yA_min, wA, hA = boxA
    xA_max, yA_max = xA_min + wA, yA_min + hA
    
    xB_min, yB_min, wB, hB = boxB
    xB_max, yB_max = xB_min + wB, yB_min + hB

    x_inter_min = max(xA_min, xB_min)
    y_inter_min = max(yA_min, yB_min)
    x_inter_max = min(xA_max, xB_max)
    y_inter_max = min(yA_max, yB_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    area_inter = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    area_A = wA * hA
    area_B = wB * hB
    
    return area_inter / float(area_A + area_B - area_inter)

def calcular_centroide(box):
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

def calcular_bounding_box_global(caixas):
    x_min = min([box[0] for box in caixas])
    y_min = min([box[1] for box in caixas])
    x_max = max([box[0] + box[2] for box in caixas])
    y_max = max([box[1] + box[3] for box in caixas])
    return x_min, y_min, x_max - x_min, y_max - y_min

def supressao_nao_maxima_inteligente(candidatos_com_dna, max_overlap=0.80):
    if not candidatos_com_dna: 
        return []
    candidatos_ordenados = sorted(candidatos_com_dna, key=lambda item: item[0][2] * item[0][3], reverse=True)
    caixas_finais = []

    for atual in candidatos_ordenados:
        box_atual, dna_atual = atual
        x1, y1, w1, h1 = box_atual
        area1 = w1 * h1
        redundante = False

        for aprovada in caixas_finais:
            box_aprovada, dna_aprovada = aprovada
            x2, y2, w2, h2 = box_aprovada
            
            x_inter_min = max(x1, x2)
            y_inter_min = max(y1, y2)
            x_inter_max = min(x1 + w1, x2 + w2)
            y_inter_max = min(y1 + h1, y2 + h2)

            if x_inter_max > x_inter_min and y_inter_max > y_inter_min:
                area_inter = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
                ioa = area_inter / float(area1)
                
                if ioa > max_overlap:
                    if dna_atual.issubset(dna_aprovada):
                        redundante = True
                        break 
        
        if not redundante:
            caixas_finais.append(atual)
            
    return [item[0] for item in caixas_finais]

# --- REMOÇÃO DE CAIXAS VERDES INTERNAS ---
def remover_caixas_internas(caixas_verdes, max_ioa=0.85):
    """
    Se uma caixa verde estiver mais de 85% dentro de outra caixa verde,
    apaga a menor e fica só com a maior.
    """
    if not caixas_verdes:
        return []
        
    caixas_ordenadas = sorted(caixas_verdes, key=lambda b: b[2] * b[3], reverse=True)
    caixas_finais = []
    
    for box_atual in caixas_ordenadas:
        x1, y1, w1, h1 = box_atual
        area1 = w1 * h1
        engolida = False
        
        for box_aprovada in caixas_finais:
            x2, y2, w2, h2 = box_aprovada
            
            x_inter_min = max(x1, x2)
            y_inter_min = max(y1, y2)
            x_inter_max = min(x1 + w1, x2 + w2)
            y_inter_max = min(y1 + h1, y2 + h2)

            if x_inter_max > x_inter_min and y_inter_max > y_inter_min:
                area_inter = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
                ioa = area_inter / float(area1)
                
                # Se a caixa menor estiver engolida pela maior
                if ioa > max_ioa:
                    engolida = True
                    break
                    
        if not engolida:
            caixas_finais.append(box_atual)
            
    return caixas_finais

def atualizar_imagem():
    global img_original_atual
    if not caminhos_imagens: return
    img_original_atual = cv2.imread(caminhos_imagens[ESTADO['idx']])
    processar_e_exibir()

def processar_e_exibir(*args):
    global tk_image
    if img_original_atual is None: return

    img_trabalho = img_original_atual.copy()
    hsv = cv2.cvtColor(img_trabalho, cv2.COLOR_BGR2HSV)
    H_img, W_img = img_trabalho.shape[:2]

    inicio_tempo = time.perf_counter()

    # --- 1. MÁSCARAS ---
    mask1 = cv2.inRange(hsv, 
                        np.array([ESTADO['h1_min'], ESTADO['s_min'], ESTADO['v_min']]), 
                        np.array([ESTADO['h1_max'], ESTADO['s_max'], ESTADO['v_max']]))
    mask2 = cv2.inRange(hsv, 
                        np.array([ESTADO['h2_min'], ESTADO['s_min'], ESTADO['v_min']]), 
                        np.array([ESTADO['h2_max'], ESTADO['s_max'], ESTADO['v_max']]))
    mask_laranja = cv2.bitwise_or(mask1, mask2)
    
    k_abert = ESTADO['k_abertura']
    if k_abert % 2 == 0: k_abert += 1 
    if ESTADO['iter_abertura'] > 0 and k_abert > 0:
        kernel_a = np.ones((k_abert, k_abert), np.uint8)
        mask_laranja_limpa = cv2.morphologyEx(mask_laranja, cv2.MORPH_OPEN, kernel_a, iterations=ESTADO['iter_abertura'])
    else:
        mask_laranja_limpa = mask_laranja

    mask_branco = cv2.inRange(hsv, np.array([0, 0, 152]), np.array([179, 74, 255]))
    mask_branco_limpa = cv2.morphologyEx(mask_branco, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_branco_limpa = cv2.morphologyEx(mask_branco_limpa, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # --- 2. ENCONTRAR BLOCOS ---
    contornos, _ = cv2.findContours(mask_laranja_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caixas_validas = []
    pixels_por_caixa = []
    centroides = []
    
    area_min_absoluta = ESTADO['area_minima']
    fator_proporcao = ESTADO['fator_proporcao']
    
    areas = [cv2.contourArea(cnt) for cnt in contornos]
    area_maxima_img = max(areas) if areas else 0
    area_min_adaptativa = area_maxima_img * fator_proporcao
    limite_area_final = max(area_min_absoluta, area_min_adaptativa)
    
    img_regioes = np.zeros_like(img_trabalho)
    idx_regiao_valida = 0
    
    for cnt in contornos:
        if cv2.contourArea(cnt) > limite_area_final:
            x, y, w, h = cv2.boundingRect(cnt)
            caixas_validas.append((x, y, w, h))
            roi = mask_laranja_limpa[y:y+h, x:x+w]
            pixels_por_caixa.append(cv2.countNonZero(roi))
            centroides.append(calcular_centroide((x, y, w, h)))
            
            np.random.seed(idx_regiao_valida)
            cv2.drawContours(img_regioes, [cnt], -1, tuple(int(c) for c in np.random.randint(60, 255, size=3)), -1)
            idx_regiao_valida += 1

    num_caixas = len(caixas_validas)
    
    # --- 3. K-VIZINHOS DINÂMICO ---
    combos_unicos = set()
    fator_raio = ESTADO['fator_raio']
    k_viz = ESTADO['k_vizinhos']
    
    for i in range(num_caixas):
        distancias_validas = []
        x_i, y_i, w_i, h_i = caixas_validas[i]
        raio_dinamico_i = max(w_i, h_i) * fator_raio
        for j in range(num_caixas):
            dist = math.hypot(centroides[i][0] - centroides[j][0], centroides[i][1] - centroides[j][1])
            if dist <= raio_dinamico_i:
                distancias_validas.append((dist, j))
        distancias_validas.sort()
        vizinhos_k = [idx for dist, idx in distancias_validas[:k_viz]]
        
        limite_comb = min(len(vizinhos_k), 4)
        for tamanho in range(1, limite_comb + 1):
            for combo in combinations(vizinhos_k, tamanho):
                combos_unicos.add(tuple(sorted(combo)))

    # --- 4. DENSIDADE E REDUNDÂNCIA ---
    l_min = ESTADO['limiar_laranja_min']
    l_max = ESTADO['limiar_laranja_max']
    t_min = ESTADO['limiar_total']
    bboxes_com_dna = {}
    
    for combo_indices in combos_unicos:
        caixas_do_combo = [caixas_validas[idx] for idx in combo_indices]
        x_g, y_g, w_g, h_g = calcular_bounding_box_global(caixas_do_combo)
        if w_g <= 5 or h_g <= 5: continue
            
        area_total_caixa = w_g * h_g
        pixels_combo_estrito = sum([pixels_por_caixa[idx] for idx in combo_indices])
        densidade_laranja = pixels_combo_estrito / float(area_total_caixa)

        roi_branco = mask_branco_limpa[y_g:y_g+h_g, x_g:x_g+w_g]
        pixels_branco_total = cv2.countNonZero(roi_branco)
        densidade_total = (pixels_combo_estrito + pixels_branco_total) / float(area_total_caixa)

        if l_min <= densidade_laranja <= l_max and densidade_total >= t_min:
            bbox = (x_g, y_g, w_g, h_g)
            conjunto_dna = set(combo_indices)
            if bbox not in bboxes_com_dna or len(conjunto_dna) > len(bboxes_com_dna[bbox]):
                bboxes_com_dna[bbox] = conjunto_dna

    candidatos_brutos = list(bboxes_com_dna.items())
    candidatos_finais = supressao_nao_maxima_inteligente(candidatos_brutos, max_overlap=0.80)
    
    # --- 5. LÓGICA DE IoU COM GABARITO ---
    img_display = img_trabalho.copy()
    caminho_img_atual = caminhos_imagens[ESTADO['idx']]
    caixas_gabarito = ler_gabarito_yolo(caminho_img_atual, W_img, H_img)

    lim_pos = ESTADO['iou_positivo']
    lim_neg = ESTADO['iou_negativo']
    
    lista_verdes = []
    lista_vermelhas = []
    lista_cinzas = []

    # Separa os candidatos pelo IoU
    for cand in candidatos_finais:
        maior_iou = 0.0
        for gab in caixas_gabarito:
            iou_atual = calcular_iou(cand, gab)
            if iou_atual > maior_iou:
                maior_iou = iou_atual
                
        if maior_iou >= lim_pos:
            lista_verdes.append(cand)
        elif maior_iou <= lim_neg:
            lista_vermelhas.append(cand)
        else:
            lista_cinzas.append(cand)

    # Aplica o novo filtro para apagar verdes redundantes internas
    lista_verdes = remover_caixas_internas(lista_verdes, max_ioa=0.85)

    # --- PINTURA EM CAMADAS ---
    # As caixas desenhadas por último ficam sobrepostas às desenhadas primeiro

    # 1. Camada de Baixo: Ignorados / Zona Cinzenta (Cinza)
    for cand in lista_cinzas:
        xc, yc, wc, hc = cand
        cv2.rectangle(img_display, (xc, yc), (xc+wc, yc+hc), (128, 128, 128), 2)

    # 2. Camada do Meio: Rejeitados / Hard Negatives (Vermelho)
    for cand in lista_vermelhas:
        xc, yc, wc, hc = cand
        cv2.rectangle(img_display, (xc, yc), (xc+wc, yc+hc), (0, 0, 255), 2)

    # 3. Camada do Topo: Aceitos / Cones (Verde)
    for cand in lista_verdes:
        xc, yc, wc, hc = cand
        cv2.rectangle(img_display, (xc, yc), (xc+wc, yc+hc), (0, 255, 0), 2)

    fim_tempo = time.perf_counter()
    tempo_ms = (fim_tempo - inicio_tempo) * 1000

    # --- 6. RENDERIZAÇÃO DOS PAINÉIS ---
    mask_laranja_bgr = cv2.cvtColor(mask_laranja_limpa, cv2.COLOR_GRAY2BGR)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    
    # Textos de Interface limpos no canto
    cv2.putText(img_display, f"Tempo: {tempo_ms:.1f} ms", (10, 30), fonte, 0.6, (255, 255, 0), 2)
    cv2.putText(img_display, f"Aceitos (Verde): {len(lista_verdes)}", (10, 60), fonte, 0.6, (0, 255, 0), 2)
    cv2.putText(img_display, f"Rejeitados (Vermelho): {len(lista_vermelhas)}", (10, 90), fonte, 0.6, (0, 0, 255), 2)
    cv2.putText(img_display, f"Ignorados (Cinza): {len(lista_cinzas)}", (10, 120), fonte, 0.6, (128, 128, 128), 2)

    cv2.putText(mask_laranja_bgr, "Mascara Laranja", (10, 30), fonte, 0.7, (0, 165, 255), 2)
    cv2.putText(img_regioes, f"Regioes base: {num_caixas}", (10, 30), fonte, 0.6, (255, 255, 255), 2)

    imagem_combinada = cv2.hconcat([img_display, mask_laranja_bgr, img_regioes])

    h_img_c, w_img_c = imagem_combinada.shape[:2]
    max_w, max_h = 1120, 800  
    escala = min(max_w / w_img_c, max_h / h_img_c)
    
    novo_largo = int(w_img_c * escala)
    nova_altura = int(h_img_c * escala)
    imagem_combinada = cv2.resize(imagem_combinada, (novo_largo, nova_altura))

    imagem_rgb = cv2.cvtColor(imagem_combinada, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    label_imagem.config(image=tk_image)

def ao_mudar_slider(chave, valor, is_float=False):
    ESTADO[chave] = float(valor) if is_float else int(float(valor))
    processar_e_exibir()

def proxima_imagem(event=None):
    ESTADO['idx'] = min(ESTADO['idx'] + 1, len(caminhos_imagens) - 1)
    slider_img.set(ESTADO['idx'])
    atualizar_imagem()

def imagem_anterior(event=None):
    ESTADO['idx'] = max(ESTADO['idx'] - 1, 0)
    slider_img.set(ESTADO['idx'])
    atualizar_imagem()

def imprimir_e_sair():
    print("\n" + "="*50)
    print(" DUMP DE PARÂMETROS OTIMIZADOS - VISÃO COMPUTACIONAL")
    print("="*50)
    print("\n## 1. Filtros de Cor Laranja (HSV)")
    print(f"limite_inferior_1 = np.array([{ESTADO['h1_min']}, {ESTADO['s_min']}, {ESTADO['v_min']}])")
    print(f"limite_superior_1 = np.array([{ESTADO['h1_max']}, {ESTADO['s_max']}, {ESTADO['v_max']}])")
    print(f"limite_inferior_2 = np.array([{ESTADO['h2_min']}, {ESTADO['s_min']}, {ESTADO['v_min']}])")
    print(f"limite_superior_2 = np.array([{ESTADO['h2_max']}, {ESTADO['s_max']}, {ESTADO['v_max']}])")
    
    print("\n## 2. Morfologia Matemática")
    print(f"kernel_abertura = np.ones(({ESTADO['k_abertura']}, {ESTADO['k_abertura']}), np.uint8)")
    print(f"iteracoes_abertura = {ESTADO['iter_abertura']}")
    
    print("\n## 3. Heurísticas de Base (Blocos)")
    print(f"area_minima_absoluta = {ESTADO['area_minima']} px")
    print(f"fator_proporcao_adaptativo = {ESTADO['fator_proporcao']:.3f}")
    
    print("\n## 4. Agrupamento Espacial (K-NN)")
    print(f"k_vizinhos = {ESTADO['k_vizinhos']}")
    print(f"fator_raio_distancia = {ESTADO['fator_raio']:.2f}")
    
    print("\n## 5. Regras de Densidade Final")
    print(f"limiar_laranja_min = {ESTADO['limiar_laranja_min']:.2f}")
    print(f"limiar_laranja_max = {ESTADO['limiar_laranja_max']:.2f}")
    print(f"limiar_total_min   = {ESTADO['limiar_total']:.2f}")
    
    print("\n## 6. Limiares de Separação de Treino (IoU)")
    print(f"LIMIAR_POSITIVO (CONE) = >= {ESTADO['iou_positivo']:.2f}")
    print(f"LIMIAR_NEGATIVO (FUNDO) = <= {ESTADO['iou_negativo']:.2f}")
    print("="*50 + "\n")
    root.destroy()

# --- Construção da Interface Gráfica ---
root = tk.Tk()
root.title("Painel Supremo: Segmentação + Filtro Espacial + Calibração IoU")
root.geometry("1550x850") 

frame_controles = tk.Frame(root, width=400, padx=10, pady=5)
frame_controles.pack(side=tk.LEFT, fill=tk.Y)

canvas = tk.Canvas(frame_controles, borderwidth=0, width=380)
scrollbar = tk.Scrollbar(frame_controles, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

frame_visualizacao = tk.Frame(root, bg="black")
frame_visualizacao.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

label_imagem = tk.Label(frame_visualizacao, bg="black")
label_imagem.pack(expand=True)

root.bind('d', proxima_imagem)
root.bind('a', imagem_anterior)

def criar_slider(frame, texto, chave, min_val, max_val, resolucao=1, is_float=False):
    lbl = tk.Label(frame, text=texto, font=("Arial", 9))
    lbl.pack(anchor='w')
    s = tk.Scale(frame, from_=min_val, to=max_val, orient='horizontal', resolution=resolucao,
                 command=lambda v, c=chave: ao_mudar_slider(c, v, is_float))
    s.set(ESTADO[chave])
    s.pack(fill='x')
    return s

total_imgs = carregar_dataset("images/train")

frame_nav = tk.LabelFrame(scrollable_frame, text="Navegação (Teclado: A / D)")
frame_nav.pack(fill='x', pady=4)
slider_img = tk.Scale(frame_nav, from_=0, to=total_imgs-1, orient='horizontal', command=lambda v: (ao_mudar_slider('idx', v, False), atualizar_imagem()))
slider_img.pack(fill='x')

frame_iou = tk.LabelFrame(scrollable_frame, text="Calibração de Dataset (IoU)", fg="purple")
frame_iou.pack(fill='x', pady=4)
criar_slider(frame_iou, "IoU POSITIVO (Cone > X)", 'iou_positivo', 0.10, 1.00, 0.01, True)
criar_slider(frame_iou, "IoU NEGATIVO (Fundo < X)", 'iou_negativo', 0.00, 0.50, 0.01, True)

frame_h1 = tk.LabelFrame(scrollable_frame, text="Cor Laranja: Filtro Principal")
frame_h1.pack(fill='x', pady=4)
criar_slider(frame_h1, "H1 Min", 'h1_min', 0, 179)
criar_slider(frame_h1, "H1 Max", 'h1_max', 0, 179)

frame_h2 = tk.LabelFrame(scrollable_frame, text="Cor Laranja: Tons Altos (Vermelhos)")
frame_h2.pack(fill='x', pady=4)
criar_slider(frame_h2, "H2 Min", 'h2_min', 0, 179)
criar_slider(frame_h2, "H2 Max", 'h2_max', 0, 179)

frame_sv = tk.LabelFrame(scrollable_frame, text="Saturação e Brilho (Laranja)")
frame_sv.pack(fill='x', pady=4)
criar_slider(frame_sv, "S Min", 's_min', 0, 255)
criar_slider(frame_sv, "V Min", 'v_min', 0, 255)

frame_morf = tk.LabelFrame(scrollable_frame, text="Limpeza de Máscara (Abertura)", fg="red")
frame_morf.pack(fill='x', pady=4)
criar_slider(frame_morf, "Tamanho do Pincel", 'k_abertura', 1, 15, resolucao=2)
criar_slider(frame_morf, "Iterações (Força)", 'iter_abertura', 0, 5)

frame_geo = tk.LabelFrame(scrollable_frame, text="Parâmetros de Pipeline", fg="blue")
frame_geo.pack(fill='x', pady=4)
criar_slider(frame_geo, "Área Mínima Absoluta", 'area_minima', 1, 50, 1, False)
criar_slider(frame_geo, "Fator Proporção (Adaptativo)", 'fator_proporcao', 0.0, 0.1, 0.001, True)
criar_slider(frame_geo, "K Vizinhos (KNN)", 'k_vizinhos', 1, 12, 1, False)
criar_slider(frame_geo, "Fator Raio (Distância)", 'fator_raio', 0.5, 5.0, 0.1, True)

frame_dens = tk.LabelFrame(scrollable_frame, text="Cortes de Densidade")
frame_dens.pack(fill='x', pady=4)
criar_slider(frame_dens, "Densidade Lar. MÍNIMA", 'limiar_laranja_min', 0.0, 1.0, 0.01, True)
criar_slider(frame_dens, "Densidade Lar. MÁXIMA", 'limiar_laranja_max', 0.0, 1.0, 0.01, True)
criar_slider(frame_dens, "Densidade TOTAL (Lar + Br)", 'limiar_total', 0.0, 1.0, 0.01, True)

btn_sair = tk.Button(scrollable_frame, text="SALVAR E SAIR", bg="green", fg="white", font=("Arial", 11, "bold"), command=imprimir_e_sair)
btn_sair.pack(fill='x', pady=10)

if total_imgs > 0:
    atualizar_imagem()
    root.mainloop()
else:
    print("Aviso: Nenhuma imagem encontrada na pasta especificada.")