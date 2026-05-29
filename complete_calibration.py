import cv2
import numpy as np
import os
import glob
import tkinter as tk
import math
import time
from itertools import combinations
from PIL import Image, ImageTk

# --- Configurações Iniciais do Estado (Parâmetros Calibrados) ---
ESTADO = {
    'idx': 0,
    # 1. Filtros de Cor Laranja (HSV)
    'h1_min': 0, 'h1_max': 19,
    'h2_min': 175, 'h2_max': 179,
    's_min': 110, 's_max': 255,
    'v_min': 75, 'v_max': 255,
    # 2. Morfologia Matemática
    'k_abertura': 3, 'iter_abertura': 1,
    # 3. Heurísticas de Base (Blocos)
    'area_minima': 5,
    'fator_proporcao': 0.015,
    # 4 e 5. Regras de Densidade e Espaciais (K-NN)
    'limiar_laranja_min': 0.20,
    'limiar_laranja_max': 0.95,
    'limiar_total': 0.30,
    'k_vizinhos': 6,
    'fator_raio': 2.20
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

def calcular_centroide(box):
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

def calcular_bounding_box_global(caixas):
    x_min = min([box[0] for box in caixas])
    y_min = min([box[1] for box in caixas])
    x_max = max([box[0] + box[2] for box in caixas])
    y_max = max([box[1] + box[3] for box in caixas])
    return x_min, y_min, x_max - x_min, y_max - y_min

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

    # --- INÍCIO DA MEDIÇÃO DE TEMPO DO PIPELINE LÓGICO ---
    inicio_tempo = time.perf_counter()

    # --- 1A. MÁSCARA LARANJA DINÂMICA ---
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

    # --- 1B. MÁSCARA BRANCA ---
    mask_branco = cv2.inRange(hsv, np.array([0, 0, 152]), np.array([179, 74, 255]))
    mask_branco_limpa = cv2.morphologyEx(mask_branco, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_branco_limpa = cv2.morphologyEx(mask_branco_limpa, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # --- 2. ENCONTRAR BLOCOS (COM ÁREA ADAPTATIVA) ---
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
            cor_unica = tuple(int(c) for c in np.random.randint(60, 255, size=3))
            cv2.drawContours(img_regioes, [cnt], -1, cor_unica, -1)
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

    # --- 4. FILTROS DE DENSIDADE ---
    l_min = ESTADO['limiar_laranja_min']
    l_max = ESTADO['limiar_laranja_max']
    t_min = ESTADO['limiar_total']
    
    candidatos_finais = []
    
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
            candidatos_finais.append((x_g, y_g, w_g, h_g))

    # --- FIM DA MEDIÇÃO DE TEMPO ---
    fim_tempo = time.perf_counter()
    tempo_ms = (fim_tempo - inicio_tempo) * 1000

    # --- 5. COMPOSIÇÃO DOS PAINÉIS ---
    img_display = img_trabalho.copy()
    for (x, y, w, h) in candidatos_finais:
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    mask_laranja_bgr = cv2.cvtColor(mask_laranja_limpa, cv2.COLOR_GRAY2BGR)

    fonte = cv2.FONT_HERSHEY_SIMPLEX
    # Exibe Finais + Tempo de processamento na imagem
    texto_finais = f"Finais: {len(candidatos_finais)} | Tempo: {tempo_ms:.1f} ms"
    cv2.putText(img_display, texto_finais, (10, 30), fonte, 0.7, (255, 255, 0), 2)
    
    cv2.putText(mask_laranja_bgr, "Mascara Laranja", (10, 30), fonte, 0.7, (0, 165, 255), 2)
    cv2.putText(img_regioes, f"Regioes base: {num_caixas} (Corte: >{int(limite_area_final)}px)", (10, 30), fonte, 0.6, (255, 255, 255), 2)

    imagem_combinada = cv2.hconcat([img_display, mask_laranja_bgr, img_regioes])

    # --- REDIMENSIONAMENTO INTELIGENTE ---
    h_img, w_img = imagem_combinada.shape[:2]
    
    max_w = 1120 
    max_h = 800  

    escala = min(max_w / w_img, max_h / h_img)
    
    novo_largo = int(w_img * escala)
    nova_altura = int(h_img * escala)
    
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
    print(f"fator_proporcao_adaptativo = {ESTADO['fator_proporcao']:.3f} ({(ESTADO['fator_proporcao']*100):.1f}% do maior objeto)")
    
    print("\n## 4. Agrupamento Espacial (K-NN)")
    print(f"k_vizinhos = {ESTADO['k_vizinhos']}")
    print(f"fator_raio_distancia = {ESTADO['fator_raio']:.2f}")
    
    print("\n## 5. Regras de Densidade Final")
    print(f"limiar_laranja_min = {ESTADO['limiar_laranja_min']:.2f}")
    print(f"limiar_laranja_max = {ESTADO['limiar_laranja_max']:.2f}")
    print(f"limiar_total_min   = {ESTADO['limiar_total']:.2f}")
    print("="*50 + "\n")
    root.destroy()

# --- Construção da Interface Gráfica ---
root = tk.Tk()
root.title("Painel Supremo de Calibração: Segmentação + Filtro Espacial")
root.geometry("1550x850") 

frame_controles = tk.Frame(root, width=400, padx=10, pady=5)
frame_controles.pack(side=tk.LEFT, fill=tk.Y)

# Canvas de Rolagem
canvas = tk.Canvas(frame_controles, borderwidth=0, width=380)
scrollbar = tk.Scrollbar(frame_controles, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
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

# Bloco de Navegação
frame_nav = tk.LabelFrame(scrollable_frame, text="Navegação (Teclado: A / D)")
frame_nav.pack(fill='x', pady=4)
slider_img = tk.Scale(frame_nav, from_=0, to=total_imgs-1, orient='horizontal', 
                      command=lambda v: (ao_mudar_slider('idx', v, False), atualizar_imagem()))
slider_img.pack(fill='x')

# Bloco HSV Laranja
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

# Morfologia
frame_morf = tk.LabelFrame(scrollable_frame, text="Limpeza de Máscara (Abertura)", fg="red")
frame_morf.pack(fill='x', pady=4)
criar_slider(frame_morf, "Tamanho do Pincel", 'k_abertura', 1, 15, resolucao=2)
criar_slider(frame_morf, "Iterações (Força)", 'iter_abertura', 0, 5)

# Geometria e Pipeline (COM ADAPTATIVO)
frame_geo = tk.LabelFrame(scrollable_frame, text="Parâmetros de Pipeline", fg="blue")
frame_geo.pack(fill='x', pady=4)
criar_slider(frame_geo, "Área Mínima Absoluta", 'area_minima', 1, 50, 1, False)
criar_slider(frame_geo, "Fator Proporção (Adaptativo)", 'fator_proporcao', 0.0, 0.1, 0.001, True)
criar_slider(frame_geo, "K Vizinhos (KNN)", 'k_vizinhos', 1, 12, 1, False)
criar_slider(frame_geo, "Fator Raio (Distância)", 'fator_raio', 0.5, 5.0, 0.1, True)

# Densidade
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