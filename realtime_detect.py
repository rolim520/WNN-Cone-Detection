import cv2
import time
import collections
import json
import wisardpkg as wp
from utils import *

ESTADO = carregar_configuracoes()

# ==========================================================
# PARÂMETROS OTIMIZADOS
# ==========================================================
# --- NOVO: Variável para definir qual modelo usar ---
TIPO_MODELO = 'wisard'  # Escolha entre: 'wisard' ou 'cluswisard'

MODO_BINARIZACAO = 'cor' # Opções: 'cor', 'canny', 'hibrido'
RESOLUCAO = 64
IGN_ZERO = False
LARGURA_WEBCAM = 640
EXIBIR_TODAS_CAIXAS = False
LIMIAR_AR_CONE = 1.25

# ==========================================================
# PARÂMETRO DE RASTREAMENTO TEMPORAL
# ==========================================================
N_FRAMES_MEMORIA = 3
LIMIAR_CONFIANCA = 0.0
# ==========================================================

print(f"\n[Fase 1] Carregando Modelo {TIPO_MODELO.upper()} salvo...")

# Aponta para a pasta gerada pelos scripts de benchmark
pasta_origem = "resultados_wisard" if TIPO_MODELO == 'wisard' else "resultados_cluswisard"
caminho_modelo = f"{pasta_origem}/melhor_modelo.json"

try:
    # Carrega a string gigante do JSON do disco
    t_load_ini = time.perf_counter()
    with open(caminho_modelo, 'r') as f:
        json_str = f.read()
    
    # Instancia o modelo reconstruindo-o pelo JSON
    if TIPO_MODELO == 'wisard':
        modelo = wp.Wisard(json_str)
    elif TIPO_MODELO == 'cluswisard':
        modelo = wp.ClusWisard(json_str)
    else:
        raise ValueError("TIPO_MODELO inválido. Use 'wisard' ou 'cluswisard'.")
        
    print(f" -> Modelo carregado com sucesso em {(time.perf_counter() - t_load_ini)*1000:.2f} ms!")

except FileNotFoundError:
    print(f"[!] ERRO: O arquivo {caminho_modelo} não foi encontrado.")
    print("Execute os scripts de benchmark primeiro para gerar o arquivo JSON.")
    exit()

print("\n[Fase 2] Abrindo a Webcam...")
cap = cv2.VideoCapture(0)
fps_medio = 0

memoria_caixas = collections.deque(maxlen=N_FRAMES_MEMORIA)

while True:
    t_inicio = time.perf_counter()
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (LARGURA_WEBCAM, int(frame.shape[0] * (LARGURA_WEBCAM / frame.shape[1]))))
    h_img, w_img = frame.shape[:2]
    
    mask_lar, mask_br = gerar_mascaras(frame, ESTADO)
    mask_canny = gerar_canny(frame, ESTADO)
    
    candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO)
    
    # Injeção da Memória
    for caixas_antigas in memoria_caixas:
        for box_antiga in caixas_antigas:
            if box_antiga not in candidatos:
                candidatos.append(box_antiga)

    caixas_filtradas = []
    
    if candidatos:
        recortes, candidatos_validos = [], []
        
        for (x, y, w, h) in candidatos:
            x, y = max(0, x), max(0, y)
            if x + w > w_img: w = w_img - x
            if y + h > h_img: h = h_img - y
            
            c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
            c_canny = mask_canny[y:y+h, x:x+w]
            
            if c_lar.size > 0:
                c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                recortes.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO, c_canny, MODO_BINARIZACAO))
                candidatos_validos.append((x, y, w, h))
                
        if recortes:
            votos_preds = modelo.rank(wp.DataSet(recortes))
            
            candidatos_com_score = []
            for i, votos_dict in enumerate(votos_preds):
                classe_predita = max(votos_dict, key=votos_dict.get)
                # No ClusWiSARD a chave vem como 'cone::0', então fatiamos o nome da classe base
                classe_base = classe_predita.split('::')[0]
                
                votos_vencedor = votos_dict[classe_predita]
                total_votos = sum(votos_dict.values())
                
                confianca = votos_vencedor / total_votos if total_votos > 0 else 0.0
                
                if classe_base == 'cone' and confianca >= LIMIAR_CONFIANCA:
                    candidatos_com_score.append({
                        'box': candidatos_validos[i],
                        'score': confianca,
                        'votos_dict': votos_dict
                    })
            
            candidatos_com_score.sort(key=lambda item: item['box'][2] * item['box'][3], reverse=True)
            for item in candidatos_com_score:
                box_raw = item['box']
                if not any(calcular_iom(box_raw, b_apr['box']) > 0.6 for b_apr in caixas_filtradas):
                    caixas_filtradas.append(item)
                    
            if EXIBIR_TODAS_CAIXAS:
                for idx, box in enumerate(candidatos_validos):
                    votos_dict = votos_preds[idx]
                    classe_box_raw = max(votos_dict, key=votos_dict.get)
                    classe_box = classe_box_raw.split('::')[0]
                    cor = (0, 255, 0) if classe_box == 'cone' else (0, 0, 255)
                    espessura = 2 if classe_box == 'cone' else 1
                    cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), cor, espessura)

    # Guarda apenas a tupla de coordenadas na memória
    boxes_para_memoria = [item['box'] for item in caixas_filtradas]
    memoria_caixas.append(boxes_para_memoria)

    if not EXIBIR_TODAS_CAIXAS:
        for item in caixas_filtradas:
            box = item['box']
            score = item['score']
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Cone {score:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    t_fim = time.perf_counter()
    fps_medio = (fps_medio * 0.9) + ((1.0 / (t_fim - t_inicio)) * 0.1)
    cv2.putText(frame, f"FPS: {fps_medio:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow('WiSARD - Real Time Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()