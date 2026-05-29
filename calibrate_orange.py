import cv2
import numpy as np
import os
import glob
import tkinter as tk
from PIL import Image, ImageTk

# --- Configurações Iniciais ---
ESTADO = {
    'idx': 0,
    'h1_min': 0, 'h1_max': 21,
    'h2_min': 175, 'h2_max': 179,
    's_min': 100, 's_max': 255,
    'v_min': 100, 'v_max': 255,
    'k_abertura': 3, 'iter_abertura': 0,
    'kernel_size': 3, 'iteracoes': 0
}

caminhos_imagens = []
img_atual = None
hsv_atual = None
tk_image = None

def carregar_dataset(pasta):
    global caminhos_imagens
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        caminhos_imagens.extend(glob.glob(os.path.join(pasta, ext)))
    caminhos_imagens.sort()
    return len(caminhos_imagens)

def atualizar_imagem():
    global img_atual, hsv_atual
    if not caminhos_imagens: return
    
    img_bruta = cv2.imread(caminhos_imagens[ESTADO['idx']])
    h, w = img_bruta.shape[:2]
    
    # Reduzimos a altura para 300px para garantir que as 3 imagens caibam na tela
    nova_altura = 300 
    novo_largo = int(w * (nova_altura / h))
    img_atual = cv2.resize(img_bruta, (novo_largo, nova_altura))
    
    hsv_atual = cv2.cvtColor(img_atual, cv2.COLOR_BGR2HSV)
    processar_e_exibir()

def processar_e_exibir(*args):
    global tk_image
    if img_atual is None: return

    # 1. Máscaras de Cor
    mask1 = cv2.inRange(hsv_atual, 
                        np.array([ESTADO['h1_min'], ESTADO['s_min'], ESTADO['v_min']]), 
                        np.array([ESTADO['h1_max'], ESTADO['s_max'], ESTADO['v_max']]))
    
    mask2 = cv2.inRange(hsv_atual, 
                        np.array([ESTADO['h2_min'], ESTADO['s_min'], ESTADO['v_min']]), 
                        np.array([ESTADO['h2_max'], ESTADO['s_max'], ESTADO['v_max']]))
    
    mask_cor = cv2.bitwise_or(mask1, mask2)

    # 2. Abertura (Apaga ruído fino do fundo)
    k_abert = ESTADO['k_abertura']
    if k_abert % 2 == 0: k_abert += 1 
    
    if ESTADO['iter_abertura'] > 0 and k_abert > 0:
        kernel_a = np.ones((k_abert, k_abert), np.uint8)
        mask_cor = cv2.morphologyEx(mask_cor, cv2.MORPH_OPEN, kernel_a, iterations=ESTADO['iter_abertura'])

    # 3. Fechamento (Preenche buracos internos)
    k_size = ESTADO['kernel_size']
    if k_size % 2 == 0: k_size += 1 
    
    if ESTADO['iteracoes'] > 0 and k_size > 0:
        kernel_f = np.ones((k_size, k_size), np.uint8)
        mask_final = cv2.morphologyEx(mask_cor, cv2.MORPH_CLOSE, kernel_f, iterations=ESTADO['iteracoes'])
    else:
        mask_final = mask_cor

    # 4. Resultado e Renderização
    resultado_bgr = cv2.bitwise_and(img_atual, img_atual, mask=mask_final)
    mask_bgr = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)

    # Adicionando rótulos de texto em cada imagem para facilitar a visualização
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cor_texto = (255, 255, 255)
    img_display = img_atual.copy()
    
    cv2.putText(img_display, "", (10, 30), fonte, 0.8, cor_texto, 2)
    cv2.putText(mask_bgr, "", (10, 30), fonte, 0.8, cor_texto, 2)
    cv2.putText(resultado_bgr, "", (10, 30), fonte, 0.8, cor_texto, 2)

    # Concatena as três imagens horizontalmente
    imagem_combinada = cv2.hconcat([img_display, mask_bgr, resultado_bgr])

    imagem_rgb = cv2.cvtColor(imagem_combinada, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    label_imagem.config(image=tk_image)

def ao_mudar_slider(chave, valor):
    ESTADO[chave] = int(valor)
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
    print("\n--- Parâmetros Finais ---")
    print("## Cores")
    print(f"limite_inferior_1 = np.array([{ESTADO['h1_min']}, {ESTADO['s_min']}, {ESTADO['v_min']}])")
    print(f"limite_superior_1 = np.array([{ESTADO['h1_max']}, {ESTADO['s_max']}, {ESTADO['v_max']}])")
    print(f"limite_inferior_2 = np.array([{ESTADO['h2_min']}, {ESTADO['s_min']}, {ESTADO['v_min']}])")
    print(f"limite_superior_2 = np.array([{ESTADO['h2_max']}, {ESTADO['s_max']}, {ESTADO['v_max']}])")
    print("\n## Morfologia")
    print(f"kernel_abertura = np.ones(({ESTADO['k_abertura']}, {ESTADO['k_abertura']}), np.uint8)")
    print(f"iteracoes_abertura = {ESTADO['iter_abertura']}")
    print(f"kernel_fechamento = np.ones(({ESTADO['kernel_size']}, {ESTADO['kernel_size']}), np.uint8)")
    print(f"iteracoes_fechamento = {ESTADO['iteracoes']}")
    root.destroy()

# --- Configuração da Interface ---
root = tk.Tk()
root.title("Painel de Controles HSV Completo")
# Aumentamos a largura da janela para acomodar as 3 imagens
root.geometry("1400x800") 

frame_controles = tk.Frame(root, width=400, padx=10, pady=5)
frame_controles.pack(side=tk.LEFT, fill=tk.Y)

frame_visualizacao = tk.Frame(root, bg="black")
frame_visualizacao.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

label_imagem = tk.Label(frame_visualizacao, bg="black")
label_imagem.pack(expand=True)

root.bind('d', proxima_imagem)
root.bind('a', imagem_anterior)

def criar_slider(frame, texto, chave, min_val, max_val, resolucao=1):
    lbl = tk.Label(frame, text=texto)
    lbl.pack(anchor='w')
    s = tk.Scale(frame, from_=min_val, to=max_val, orient='horizontal', resolution=resolucao,
                 command=lambda v, c=chave: ao_mudar_slider(c, v))
    s.set(ESTADO[chave])
    s.pack(fill='x')
    return s

total_imgs = carregar_dataset("images/train")

frame_nav = tk.LabelFrame(frame_controles, text="Navegação")
frame_nav.pack(fill='x', pady=2)
slider_img = tk.Scale(frame_nav, from_=0, to=total_imgs-1, orient='horizontal', 
                      command=lambda v: (ao_mudar_slider('idx', v), atualizar_imagem()))
slider_img.pack(fill='x')

frame_h1 = tk.LabelFrame(frame_controles, text="Filtro 1: Laranja/Amarelo")
frame_h1.pack(fill='x', pady=2)
criar_slider(frame_h1, "H1 Min", 'h1_min', 0, 179)
criar_slider(frame_h1, "H1 Max", 'h1_max', 0, 179)

frame_h2 = tk.LabelFrame(frame_controles, text="Filtro 2: Vermelhos Altos")
frame_h2.pack(fill='x', pady=2)
criar_slider(frame_h2, "H2 Min", 'h2_min', 0, 179)
criar_slider(frame_h2, "H2 Max", 'h2_max', 0, 179)

frame_sv = tk.LabelFrame(frame_controles, text="Saturação e Brilho")
frame_sv.pack(fill='x', pady=2)
criar_slider(frame_sv, "S Min", 's_min', 0, 255)
criar_slider(frame_sv, "V Min", 'v_min', 0, 255)

frame_abertura = tk.LabelFrame(frame_controles, text="Limpeza de Ruído Externo (Abertura)", fg="red")
frame_abertura.pack(fill='x', pady=2)
criar_slider(frame_abertura, "Tamanho do Pincel", 'k_abertura', 1, 15, resolucao=2)
criar_slider(frame_abertura, "Iterações (Força)", 'iter_abertura', 0, 5)

frame_fechamento = tk.LabelFrame(frame_controles, text="Preenchimento de Falhas (Fechamento)", fg="blue")
frame_fechamento.pack(fill='x', pady=2)
criar_slider(frame_fechamento, "Tamanho do Pincel", 'kernel_size', 1, 31, resolucao=2)
criar_slider(frame_fechamento, "Iterações (Força)", 'iteracoes', 0, 10)

btn_sair = tk.Button(frame_controles, text="SALVAR E SAIR", bg="green", fg="white", command=imprimir_e_sair)
btn_sair.pack(fill='x', pady=5)

if total_imgs > 0:
    atualizar_imagem()
    root.mainloop()