import cv2
import time
import glob
import random
import numpy as np
import collections
import wisardpkg as wp
from utils import *

ESTADO = carregar_configuracoes()

# ==========================================================
# PARÂMETROS OTIMIZADOS
# ==========================================================
MODO_BINARIZACAO = 'cor' # Opções: 'cor', 'canny', 'hibrido'
RESOLUCAO = 64
TUPLA = 16
IGN_ZERO = False
LARGURA_WEBCAM = 640
EXIBIR_TODAS_CAIXAS = False
LIMIAR_AR_CONE = 1.25

# ==========================================================
# PARÂMETRO DE RASTREAMENTO TEMPORAL
# ==========================================================
# Quantos frames no passado o sistema deve lembrar?
N_FRAMES_MEMORIA = 3

# Limiar de confiança para validar a detecção (0.0 a 1.0)
LIMIAR_CONFIANCA = 0.0
# ==========================================================

print(f"\n[Fase 1] Treinando WiSARD para Tempo Real (Modo: {MODO_BINARIZACAO.upper()})...")
cones_X, fundos_X = [], []
for arq in glob.glob("images/train/*.*"):
    img = cv2.imread(arq)
    if img is None: continue
    h_img, w_img = img.shape[:2]
    gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
    
    mask_lar, mask_br = gerar_mascaras(img, ESTADO)
    mask_canny = gerar_canny(img, ESTADO)
    
    # 1. Extração de Gabaritos
    for x, y, w, h in gabaritos:
        x, y = max(0, x), max(0, y)
        c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
        c_canny = mask_canny[y:y+h, x:x+w]
        
        if c_lar.size > 0:
            c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
            for v_lar, v_br, v_canny in augmentar_recorte_mascaras(c_lar, c_br, c_canny):
                cones_X.append(binarizar_para_resolucao(v_lar, v_br, RESOLUCAO, v_canny, MODO_BINARIZACAO))
                
    # 2. Extração de Candidatos para Treino (Fundos e Cones)
    for cand in extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO):
        x, y, w, h = cand
        x, y = max(0, x), max(0, y)
        c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
        c_canny = mask_canny[y:y+h, x:x+w]
        
        if c_lar.size == 0: continue
        
        iou = max([calcular_iou(cand, gab) for gab in gabaritos], default=0.0)
        
        if iou <= ESTADO['iou_negativo']:
            c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
            fundos_X.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO, c_canny, MODO_BINARIZACAO))
        elif iou >= ESTADO['iou_positivo']:
            c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
            cones_X.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO, c_canny, MODO_BINARIZACAO))

random.seed(42)
random.shuffle(fundos_X)
fundos_X = fundos_X[:len(cones_X)]
modelo = wp.Wisard(TUPLA, ignoreZero=IGN_ZERO, returnConfidence=True)
modelo.train(wp.DataSet(cones_X + fundos_X, ['cone']*len(cones_X) + ['nao_cone']*len(fundos_X)))

print("\n[Fase 2] Abrindo a Webcam...")
cap = cv2.VideoCapture(0)
fps_medio = 0

# Fila de memória para guardar os cones aprovados nos últimos N frames
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
    
    # ==========================================================
    # INJEÇÃO DE MEMÓRIA (TRACKING)
    # ==========================================================
    for caixas_antigas in memoria_caixas:
        for box_antiga in caixas_antigas:
            # box_antiga já é apenas a tupla (x,y,w,h)
            if box_antiga not in candidatos:
                candidatos.append(box_antiga)
    # ==========================================================

    caixas_filtradas = []
    
    if candidatos:
        recortes, candidatos_validos = [], []
        
        # 3. Extração da Câmera em Tempo Real
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
            # <-- MUDANÇA AQUI: Usando modelo.rank() em vez de modelo.classify()
            votos_preds = modelo.rank(wp.DataSet(recortes))
            
            candidatos_com_score = []
            for i, votos_dict in enumerate(votos_preds):
                classe_predita = max(votos_dict, key=votos_dict.get)
                
                votos_vencedor = votos_dict[classe_predita]
                total_votos = sum(votos_dict.values())
                
                confianca = votos_vencedor / total_votos if total_votos > 0 else 0.0
                
                if classe_predita == 'cone' and confianca >= LIMIAR_CONFIANCA:
                    candidatos_com_score.append({
                        'box': candidatos_validos[i],
                        'score': confianca
                    })
            
            # Filtro NMS Corrigido para Caixas Aninhadas (agora trabalhando com dicionários)
            candidatos_com_score.sort(key=lambda item: item['box'][2] * item['box'][3], reverse=True)
            for item in candidatos_com_score:
                box_raw = item['box']
                if not any(calcular_iom(box_raw, b_apr['box']) > 0.6 for b_apr in caixas_filtradas):
                    caixas_filtradas.append(item)
                    
            if EXIBIR_TODAS_CAIXAS:
                for idx, box in enumerate(candidatos_validos):
                    votos_dict = votos_preds[idx]
                    classe_box = max(votos_dict, key=votos_dict.get)
                    cor = (0, 255, 0) if classe_box == 'cone' else (0, 0, 255)
                    espessura = 2 if classe_box == 'cone' else 1
                    cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), cor, espessura)

    # ==========================================================
    # ATUALIZAÇÃO DA MEMÓRIA
    # ==========================================================
    # Guarda apenas a tupla de coordenadas (box) na memória, ignorando o score
    boxes_para_memoria = [item['box'] for item in caixas_filtradas]
    memoria_caixas.append(boxes_para_memoria)
    # ==========================================================

    if not EXIBIR_TODAS_CAIXAS:
        for item in caixas_filtradas:
            box = item['box']
            score = item['score']
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
            # <-- MUDANÇA AQUI: Inserindo a confiança formatada com 2 casas decimais no putText
            cv2.putText(frame, f"Cone {score:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    t_fim = time.perf_counter()
    fps_medio = (fps_medio * 0.9) + ((1.0 / (t_fim - t_inicio)) * 0.1)
    cv2.putText(frame, f"FPS: {fps_medio:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow('WiSARD - Real Time Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()