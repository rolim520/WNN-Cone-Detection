import cv2
import numpy as np
import os
import glob
import tkinter as tk
import math
from itertools import combinations
from PIL import Image, ImageTk

# --- Configurações Iniciais do Estado ---
ESTADO = {
    'idx': 0,
    'area_minima': 5,            # Novo parâmetro para salvar cones no fundo!
    'limiar_laranja_min': 0.25,
    'limiar_laranja_max': 0.70,
    'limiar_total': 0.35,
    'k_vizinhos': 6,
    'fator_raio': 2.8
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

    # --- 1. PREPARAÇÃO ---
    img_trabalho = img_original_atual.copy()
    hsv = cv2.cvtColor(img_trabalho, cv2.COLOR_BGR2HSV)

    # Máscara Laranja
    mask1 = cv2.inRange(hsv, np.array([0, 87, 100]), np.array([21, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([177, 87, 100]), np.array([179, 255, 255]))
    mask_laranja = cv2.bitwise_or(mask1, mask2)
    mask_laranja_limpa = cv2.morphologyEx(mask_laranja, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Máscara Branca
    mask_branco = cv2.inRange(hsv, np.array([0, 0, 152]), np.array([179, 74, 255]))
    mask_branco_limpa = cv2.morphologyEx(mask_branco, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_branco_limpa = cv2.morphologyEx(mask_branco_limpa, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # --- 2. ENCONTRAR BLOCOS ---
    contornos, _ = cv2.findContours(mask_laranja_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caixas_validas = []
    pixels_por_caixa = []
    centroides = []
    
    area_min = ESTADO['area_minima']
    
    for cnt in contornos:
        if cv2.contourArea(cnt) > area_min: # Usando o parâmetro da interface!
            x, y, w, h = cv2.boundingRect(cnt)
            caixas_validas.append((x, y, w, h))
            roi = mask_laranja_limpa[y:y+h, x:x+w]
            pixels_por_caixa.append(cv2.countNonZero(roi))
            centroides.append(calcular_centroide((x, y, w, h)))

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
            candidatos_finais.append((x_g, y_g, w_g, h_g, densidade_laranja, densidade_total))

    # --- 5. RENDERIZAÇÃO NA TELA ---
    img_display = img_trabalho.copy()
    
    # Desenha as caixas aprovadas
    for (x, y, w, h, d_lar, d_tot) in candidatos_finais:
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Opcional: Adiciona um textinho pequeno com as métricas em cima da caixa
        # texto = f"L:{d_lar:.2f} T:{d_tot:.2f}"
        # cv2.putText(img_display, texto, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Adiciona Info Geral no canto
    info_texto = f"Caixas base: {num_caixas} | Candidatos: {len(candidatos_finais)}"
    cv2.putText(img_display, info_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Redimensiona para caber bem na tela (altura máxima 700)
    h_img, w_img = img_display.shape[:2]
    nova_altura = 700
    if h_img > nova_altura:
        novo_largo = int(w_img * (nova_altura / h_img))
        img_display = cv2.resize(img_display, (novo_largo, nova_altura))

    imagem_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    label_imagem.config(image=tk_image)

# --- Eventos da Interface ---
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
    print("\n--- Parâmetros Finais Otimizados ---")
    print(f"area_minima = {ESTADO['area_minima']}")
    print(f"limiar_laranja_min = {ESTADO['limiar_laranja_min']:.2f}")
    print(f"limiar_laranja_max = {ESTADO['limiar_laranja_max']:.2f}")
    print(f"limiar_total = {ESTADO['limiar_total']:.2f}")
    print(f"k_vizinhos = {ESTADO['k_vizinhos']}")
    print(f"fator_raio = {ESTADO['fator_raio']:.1f}")
    root.destroy()

# --- Construção da Interface ---
root = tk.Tk()
root.title("Painel de Detecção - Parâmetros do Algoritmo")
root.geometry("1200x800") 

frame_controles = tk.Frame(root, width=350, padx=10, pady=5)
frame_controles.pack(side=tk.LEFT, fill=tk.Y)

frame_visualizacao = tk.Frame(root, bg="black")
frame_visualizacao.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

label_imagem = tk.Label(frame_visualizacao, bg="black")
label_imagem.pack(expand=True)

root.bind('d', proxima_imagem)
root.bind('a', imagem_anterior)

def criar_slider(frame, texto, chave, min_val, max_val, resolucao=1, is_float=False):
    lbl = tk.Label(frame, text=texto)
    lbl.pack(anchor='w')
    s = tk.Scale(frame, from_=min_val, to=max_val, orient='horizontal', resolution=resolucao,
                 command=lambda v, c=chave: ao_mudar_slider(c, v, is_float))
    s.set(ESTADO[chave])
    s.pack(fill='x')
    return s

# --- Sliders ---
total_imgs = carregar_dataset("images/train")

frame_nav = tk.LabelFrame(frame_controles, text="Navegação (A = Voltar, D = Avançar)")
frame_nav.pack(fill='x', pady=5)
slider_img = tk.Scale(frame_nav, from_=0, to=total_imgs-1, orient='horizontal', 
                      command=lambda v: (ao_mudar_slider('idx', v, False), atualizar_imagem()))
slider_img.pack(fill='x')

frame_base = tk.LabelFrame(frame_controles, text="1. Filtro Inicial")
frame_base.pack(fill='x', pady=5)
criar_slider(frame_base, "Área Mínima (Fundo da Imagem)", 'area_minima', 1, 50, 1, False)

frame_knn = tk.LabelFrame(frame_controles, text="2. Agrupamento Espacial (K-NN)")
frame_knn.pack(fill='x', pady=5)
criar_slider(frame_knn, "K Vizinhos", 'k_vizinhos', 1, 12, 1, False)
criar_slider(frame_knn, "Fator Raio (Multiplicador)", 'fator_raio', 0.5, 5.0, 0.1, True)

frame_densidade = tk.LabelFrame(frame_controles, text="3. Filtros de Densidade")
frame_densidade.pack(fill='x', pady=5)
criar_slider(frame_densidade, "Densidade Laranja MÍNIMA", 'limiar_laranja_min', 0.0, 1.0, 0.01, True)
criar_slider(frame_densidade, "Densidade Laranja MÁXIMA", 'limiar_laranja_max', 0.0, 1.0, 0.01, True)
criar_slider(frame_densidade, "Densidade TOTAL (Lar + Branco)", 'limiar_total', 0.0, 1.0, 0.01, True)

btn_sair = tk.Button(frame_controles, text="SALVAR E SAIR", bg="green", fg="white", font=("Arial", 12, "bold"), command=imprimir_e_sair)
btn_sair.pack(fill='x', pady=15)

if total_imgs > 0:
    atualizar_imagem()
    root.mainloop()
else:
    print("Nenhuma imagem encontrada na pasta 'images/train'.")