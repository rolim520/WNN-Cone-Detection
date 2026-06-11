import cv2
import numpy as np
import os
import glob
import tkinter as tk
import time
import json
from PIL import Image, ImageTk

# Importa TODAS as funções do seu utils.py para manter a consistência do projeto!
from utils import *

ARQUIVO_CONFIG = "config.json"

def salvar_configuracoes():
    try:
        with open(ARQUIVO_CONFIG, 'w') as f:
            json.dump(ESTADO, f, indent=4)
        print(f"[*] Configurações salvas com sucesso em {ARQUIVO_CONFIG}")
    except Exception as e:
        print(f"[!] Erro ao salvar {ARQUIVO_CONFIG}: {e}")

# Carrega o estado global (a função já existe no seu utils.py)
ESTADO = carregar_configuracoes()

# Inicializa parâmetros novos e do Canny caso não existam no config.json antigo
if 'canny_limiar1' not in ESTADO: ESTADO['canny_limiar1'] = 100
if 'canny_limiar2' not in ESTADO: ESTADO['canny_limiar2'] = 200

# Novas chaves para os botões de visualização
if 'mostrar_legendas' not in ESTADO: ESTADO['mostrar_legendas'] = True
if 'sobrepor_cores' not in ESTADO: ESTADO['sobrepor_cores'] = True
if 'mostrar_laranja' not in ESTADO: ESTADO['mostrar_laranja'] = True
if 'mostrar_branca' not in ESTADO: ESTADO['mostrar_branca'] = True

caminhos_imagens = []
img_original_atual = None
tk_image = None
resize_timer = None # Timer para evitar lag ao redimensionar a janela

def carregar_dataset(pasta):
    global caminhos_imagens
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        caminhos_imagens.extend(glob.glob(os.path.join(pasta, ext)))
    caminhos_imagens.sort()
    return len(caminhos_imagens)

def atualizar_imagem():
    global img_original_atual
    if not caminhos_imagens: return
    
    img_original_atual = cv2.imread(caminhos_imagens[ESTADO['idx']])
    processar_e_exibir()

def processar_e_exibir(*args):
    global tk_image
    if img_original_atual is None: return

    img_trabalho = img_original_atual.copy()
    h_img, w_img = img_trabalho.shape[:2]

    # --- INÍCIO DA MEDIÇÃO DE TEMPO ---
    inicio_tempo = time.perf_counter()

    # 1. Usa a função unificada do utils.py para gerar ambas as máscaras
    mask_laranja, mask_branco = gerar_mascaras(img_trabalho, ESTADO)

    # 2. Usa a função unificada do utils.py para extrair os candidatos finais
    candidatos_finais = extrair_candidatos_multiplos(mask_laranja, mask_branco, w_img, h_img, ESTADO)

    # 3. Aplica a Detecção de Bordas de Canny
    edges = gerar_canny(img_trabalho, ESTADO)
    img_canny = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # --- FIM DA MEDIÇÃO DE TEMPO ---
    fim_tempo = time.perf_counter()
    tempo_ms = (fim_tempo - inicio_tempo) * 1000
    
    # Atualiza as informações de tempo e boxes no TÍTULO da janela
    root.title(f"Calibrador | Finais gerados: {len(candidatos_finais)} | Tempo de BBox: {tempo_ms:.1f} ms")

    # --- COMPOSIÇÃO DOS PAINÉIS ---

    # PAINEL 1: Display Original com Bounding Boxes
    img_display = img_trabalho.copy()
    for (x, y, w, h) in candidatos_finais:
        cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # PAINEL 2: Máscaras Combinadas (Lógica de exibição via Toggles)
    mask_combinada = np.zeros((h_img, w_img), dtype=np.uint8)
    if ESTADO['mostrar_laranja']:
        mask_combinada = cv2.bitwise_or(mask_combinada, mask_laranja)
    if ESTADO['mostrar_branca']:
        mask_combinada = cv2.bitwise_or(mask_combinada, mask_branco)

    if ESTADO['sobrepor_cores']:
        painel_mascaras = cv2.bitwise_and(img_trabalho, img_trabalho, mask=mask_combinada)
    else:
        # Se não for sobrepor cor, converte a mascara PB para 3 canais
        painel_mascaras = cv2.cvtColor(mask_combinada, cv2.COLOR_GRAY2BGR)

    # PAINEL 3: Regiões Base (Visualização Didática dos Contornos)
    img_regioes = np.zeros_like(img_trabalho)
    contornos, _ = cv2.findContours(mask_laranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = [cv2.contourArea(cnt) for cnt in contornos]
    area_maxima_img = max(areas) if areas else 0
    limite_area_final = max(ESTADO['area_minima'], area_maxima_img * ESTADO['fator_proporcao'])
    
    idx_regiao_valida = 0
    for cnt in contornos:
        if cv2.contourArea(cnt) > limite_area_final:
            np.random.seed(idx_regiao_valida)
            cor_unica = tuple(int(c) for c in np.random.randint(60, 255, size=3))
            cv2.drawContours(img_regioes, [cnt], -1, cor_unica, -1)
            idx_regiao_valida += 1

    # Textos Informativos (Controlados pelo Toggle)
    if ESTADO['mostrar_legendas']:
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        
        # Cria a string de texto com base no que está ativado
        txt_mask = "Mascaras:"
        if ESTADO['mostrar_laranja']: txt_mask += " Laranja"
        if ESTADO['mostrar_laranja'] and ESTADO['mostrar_branca']: txt_mask += " + "
        if ESTADO['mostrar_branca']: txt_mask += " Branco"
        if not ESTADO['mostrar_laranja'] and not ESTADO['mostrar_branca']: txt_mask = "Mascaras: Nenhuma"
        
        cv2.putText(painel_mascaras, txt_mask, (10, 30), fonte, 0.7, (255, 255, 255), 2)
        cv2.putText(img_regioes, f"Regioes base: {idx_regiao_valida} (Corte: >{int(limite_area_final)}px)", (10, 30), fonte, 0.6, (255, 255, 255), 2)
        cv2.putText(img_canny, f"Canny Edges ({int(ESTADO['canny_limiar1'])} - {int(ESTADO['canny_limiar2'])})", (10, 30), fonte, 0.7, (255, 255, 255), 2)

    # Montagem Final Grid 2x2
    linha1 = cv2.hconcat([img_display, painel_mascaras])
    linha2 = cv2.hconcat([img_regioes, img_canny])
    imagem_combinada = cv2.vconcat([linha1, linha2])

    # Redimensionamento adaptativo para caber na tela sem passar do original
    h_img_c, w_img_c = imagem_combinada.shape[:2]
    
    # Pega o tamanho real do frame na interface
    max_w = frame_visualizacao.winfo_width()
    max_h = frame_visualizacao.winfo_height()
    
    # Fallbacks caso a interface ainda esteja iniciando (tamanhos muito pequenos)
    if max_w < 100: max_w = 1120
    if max_h < 100: max_h = 800  

    escala = min(max_w / w_img_c, max_h / h_img_c)
    escala = min(escala, 1.0) # GARANTE que nunca vai esticar além da resolução original
    
    novo_largo = int(w_img_c * escala)
    nova_altura = int(h_img_c * escala)
    
    if novo_largo > 0 and nova_altura > 0:
        imagem_combinada = cv2.resize(imagem_combinada, (novo_largo, nova_altura))

    # Exibição no Tkinter
    imagem_rgb = cv2.cvtColor(imagem_combinada, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    label_imagem.config(image=tk_image)

def ao_mudar_slider(chave, valor, is_float=False):
    ESTADO[chave] = float(valor) if is_float else int(float(valor))
    processar_e_exibir()

def ao_mudar_checkbox(chave, var):
    ESTADO[chave] = var.get()
    processar_e_exibir()

def proxima_imagem(event=None):
    ESTADO['idx'] = min(ESTADO['idx'] + 1, len(caminhos_imagens) - 1)
    slider_img.set(ESTADO['idx'])
    atualizar_imagem()

def imagem_anterior(event=None):
    ESTADO['idx'] = max(ESTADO['idx'] - 1, 0)
    slider_img.set(ESTADO['idx'])
    atualizar_imagem()

def ao_redimensionar(event):
    global resize_timer
    # Filtra para executar apenas quando o frame_visualizacao for redimensionado
    if event.widget == frame_visualizacao:
        if resize_timer:
            root.after_cancel(resize_timer)
        # Espera 100ms depois que parou de arrastar a janela para processar a imagem
        resize_timer = root.after(100, processar_e_exibir)

def imprimir_e_sair():
    print("\n" + "="*50)
    print(" DUMP DE PARÂMETROS OTIMIZADOS - VISÃO COMPUTACIONAL")
    print("="*50)
    print("\n## 1. Filtros de Cor Laranja (HSV)")
    print(f"H1 Min: {ESTADO['h1_min']} | H1 Max: {ESTADO['h1_max']}")
    print(f"H2 Min: {ESTADO['h2_min']} | H2 Max: {ESTADO['h2_max']}")
    print(f"S Min: {ESTADO['s_min']} | S Max: {ESTADO.get('s_max', 255)}")
    print(f"V Min: {ESTADO['v_min']} | V Max: {ESTADO.get('v_max', 255)}")
    print(f"Morfologia Laranja -> Pincel Abertura: {ESTADO['k_abertura']} | Iterações: {ESTADO['iter_abertura']}")
    
    print("\n## 2. Filtros de Cor Branca (HSV)")
    print(f"H Min: {ESTADO['h_b_min']} | H Max: {ESTADO['h_b_max']}")
    print(f"S Min: {ESTADO['s_b_min']} | S Max: {ESTADO['s_b_max']}")
    print(f"V Min: {ESTADO['v_b_min']} | V Max: {ESTADO['v_b_max']}")
    print(f"Morfologia Branca -> Pincel Abertura: {ESTADO['k_abert_b']} | Iterações: {ESTADO['iter_abert_b']}")
    print(f"Morfologia Branca -> Pincel Fechamento: {ESTADO['k_fech_b']} | Iterações: {ESTADO['iter_fech_b']}")

    print("\n## 3. Pipeline e Heurísticas")
    print(f"Área Mínima Absoluta = {ESTADO['area_minima']} px")
    print(f"Fator Proporção Adaptativo = {ESTADO['fator_proporcao']:.3f}")
    print(f"K Vizinhos (KNN) = {ESTADO['k_vizinhos']}")
    print(f"Fator Raio Distância = {ESTADO['fator_raio']:.2f}")
    print(f"Razão Máx. Aspecto = {ESTADO['razao_aspecto_max']:.2f}")
    print(f"Máx. Regiões Internas = {ESTADO['max_regioes_internas']}")
    print(f"Densidade Laranja = {ESTADO['limiar_laranja_min']:.2f} a {ESTADO['limiar_laranja_max']:.2f}")
    print(f"Densidade Total (Lar + Br) = {ESTADO['limiar_total']:.2f}")

    print("\n## 4. Binarização Canny Edges")
    print(f"Canny Limiar 1 = {ESTADO['canny_limiar1']}")
    print(f"Canny Limiar 2 = {ESTADO['canny_limiar2']}")
    print("="*50 + "\n")
    
    salvar_configuracoes()
    root.destroy()

# --- Construção da Interface Gráfica ---
root = tk.Tk()
root.title("Carregando...")
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
frame_visualizacao.bind("<Configure>", ao_redimensionar) # Bind dinâmico de tela

label_imagem = tk.Label(frame_visualizacao, bg="black")
label_imagem.pack(expand=True)

root.bind('d', proxima_imagem)
root.bind('a', imagem_anterior)

def criar_slider(frame, texto, chave, min_val, max_val, resolucao=1, is_float=False):
    lbl = tk.Label(frame, text=texto, font=("Arial", 9))
    lbl.pack(anchor='w')
    s = tk.Scale(frame, from_=min_val, to=max_val, orient='horizontal', resolution=resolucao,
                 command=lambda v, c=chave: ao_mudar_slider(c, v, is_float))
    if chave in ESTADO:
        s.set(ESTADO[chave])
    s.pack(fill='x')
    return s

def criar_checkbox(frame, texto, chave):
    var = tk.BooleanVar(value=ESTADO.get(chave, True))
    cb = tk.Checkbutton(frame, text=texto, variable=var, font=("Arial", 9),
                        command=lambda: ao_mudar_checkbox(chave, var))
    cb.pack(anchor='w')
    return cb

total_imgs = carregar_dataset("images/train")

# =========================================================
# MENUS DE CALIBRAÇÃO LADO ESQUERDO
# =========================================================
frame_nav = tk.LabelFrame(scrollable_frame, text="Navegação (Teclado: A / D)")
frame_nav.pack(fill='x', pady=4)
slider_img = tk.Scale(frame_nav, from_=0, to=total_imgs-1, orient='horizontal', 
                      command=lambda v: (ao_mudar_slider('idx', v, False), atualizar_imagem()))
slider_img.set(ESTADO['idx'])
slider_img.pack(fill='x')

# --- OPÇÕES DE VISUALIZAÇÃO (NOVOS TOGGLES) ---
frame_opcoes = tk.LabelFrame(scrollable_frame, text="Opções de Visualização", fg="black")
frame_opcoes.pack(fill='x', pady=4)
criar_checkbox(frame_opcoes, "Mostrar Legendas nas Imagens", 'mostrar_legendas')
criar_checkbox(frame_opcoes, "Sobrepor Cores na Máscara", 'sobrepor_cores')
criar_checkbox(frame_opcoes, "Exibir Máscara Laranja", 'mostrar_laranja')
criar_checkbox(frame_opcoes, "Exibir Máscara Branca", 'mostrar_branca')

# --- MÁSCARA LARANJA ---
frame_h1 = tk.LabelFrame(scrollable_frame, text="Cor Laranja: Filtro Principal", fg="#d35400")
frame_h1.pack(fill='x', pady=4)
criar_slider(frame_h1, "H1 Min", 'h1_min', 0, 179)
criar_slider(frame_h1, "H1 Max", 'h1_max', 0, 179)

frame_h2 = tk.LabelFrame(scrollable_frame, text="Cor Laranja: Tons Vermelhos", fg="#d35400")
frame_h2.pack(fill='x', pady=4)
criar_slider(frame_h2, "H2 Min", 'h2_min', 0, 179)
criar_slider(frame_h2, "H2 Max", 'h2_max', 0, 179)

frame_sv = tk.LabelFrame(scrollable_frame, text="Saturação e Brilho (Laranja)", fg="#d35400")
frame_sv.pack(fill='x', pady=4)
criar_slider(frame_sv, "S Min", 's_min', 0, 255)
criar_slider(frame_sv, "V Min", 'v_min', 0, 255)

frame_morf = tk.LabelFrame(scrollable_frame, text="Morfologia (Laranja)", fg="#d35400")
frame_morf.pack(fill='x', pady=4)
criar_slider(frame_morf, "Pincel de Abertura", 'k_abertura', 1, 15, resolucao=2)
criar_slider(frame_morf, "Iterações", 'iter_abertura', 0, 5)

# --- MÁSCARA BRANCA ---
frame_hb = tk.LabelFrame(scrollable_frame, text="Cor Branca (Faixa Refletiva)", fg="#555555")
frame_hb.pack(fill='x', pady=4)
criar_slider(frame_hb, "H Min", 'h_b_min', 0, 179)
criar_slider(frame_hb, "H Max", 'h_b_max', 0, 179)
criar_slider(frame_hb, "S Min", 's_b_min', 0, 255)
criar_slider(frame_hb, "S Max", 's_b_max', 0, 255)
criar_slider(frame_hb, "V Min", 'v_b_min', 0, 255)
criar_slider(frame_hb, "V Max", 'v_b_max', 0, 255)

frame_morf_b = tk.LabelFrame(scrollable_frame, text="Morfologia (Branca)", fg="#555555")
frame_morf_b.pack(fill='x', pady=4)
criar_slider(frame_morf_b, "Abertura (Pincel)", 'k_abert_b', 1, 15, resolucao=2)
criar_slider(frame_morf_b, "Abertura (Iter)", 'iter_abert_b', 0, 5)
criar_slider(frame_morf_b, "Fechamento (Pincel)", 'k_fech_b', 1, 15, resolucao=2)
criar_slider(frame_morf_b, "Fechamento (Iter)", 'iter_fech_b', 0, 5)

# --- PIPELINE / GEOMETRIA ---
frame_geo = tk.LabelFrame(scrollable_frame, text="Parâmetros de Pipeline", fg="blue")
frame_geo.pack(fill='x', pady=4)
criar_slider(frame_geo, "Área Mínima Absoluta", 'area_minima', 1, 50, 1, False)
criar_slider(frame_geo, "Fator Proporção (Adaptativo)", 'fator_proporcao', 0.0, 0.1, 0.001, True)
criar_slider(frame_geo, "K Vizinhos (KNN)", 'k_vizinhos', 1, 12, 1, False)
criar_slider(frame_geo, "Fator Raio (Distância)", 'fator_raio', 0.5, 5.0, 0.1, True)
criar_slider(frame_geo, "Razão Máx. Aspecto", 'razao_aspecto_max', 1.0, 10.0, 0.1, True)
criar_slider(frame_geo, "Máx. Regiões na BBox", 'max_regioes_internas', 1, 20, 1, False)

# --- DENSIDADE ---
frame_dens = tk.LabelFrame(scrollable_frame, text="Cortes de Densidade", fg="blue")
frame_dens.pack(fill='x', pady=4)
criar_slider(frame_dens, "Densidade Lar. MÍNIMA", 'limiar_laranja_min', 0.0, 1.0, 0.01, True)
criar_slider(frame_dens, "Densidade Lar. MÁXIMA", 'limiar_laranja_max', 0.0, 1.0, 0.01, True)
criar_slider(frame_dens, "Densidade TOTAL (Lar + Br)", 'limiar_total', 0.0, 1.0, 0.01, True)

# --- NOVO: BINARIZAÇÃO CANNY ---
frame_canny = tk.LabelFrame(scrollable_frame, text="Binarização: Canny Edges", fg="purple")
frame_canny.pack(fill='x', pady=4)
criar_slider(frame_canny, "Limiar 1 (MinVal)", 'canny_limiar1', 0, 500, 1, False)
criar_slider(frame_canny, "Limiar 2 (MaxVal)", 'canny_limiar2', 0, 500, 1, False)

btn_sair = tk.Button(scrollable_frame, text="SALVAR E SAIR", bg="green", fg="white", font=("Arial", 11, "bold"), command=imprimir_e_sair)
btn_sair.pack(fill='x', pady=10)

if total_imgs > 0:
    # Chama explicitamente o primeiro processamento após instanciar tudo
    root.after(100, atualizar_imagem)
    root.mainloop()
else:
    print("Aviso: Nenhuma imagem encontrada na pasta especificada.")