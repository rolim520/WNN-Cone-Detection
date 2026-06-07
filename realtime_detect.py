import cv2
import time
import glob
import random
import numpy as np
import collections
import wisardpkg as wp
from utils import *

ESTADO = carregar_configuracoes()

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
# ==========================================================

print("\n[Fase 1] Treinando WiSARD para Tempo Real...")
cones_X, fundos_X = [], []
for arq in glob.glob("images/train/*.*"):
    img = cv2.imread(arq)
    if img is None: continue
    h_img, w_img = img.shape[:2]
    gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
    mask_lar, mask_br = gerar_mascaras(img, ESTADO)
    
    # 1. Extração de Gabaritos
    for x, y, w, h in gabaritos:
        c_lar, c_br = mask_lar[max(0,y):y+h, max(0,x):x+w], mask_br[max(0,y):y+h, max(0,x):x+w]
        if c_lar.size > 0:
            c_lar, c_br = alinhar_cone_vertical(c_lar, c_br, limiar_ar=LIMIAR_AR_CONE) # <-- NOVO
            for v_lar, v_br in augmentar_recorte_mascaras(c_lar, c_br):
                cones_X.append(binarizar_para_resolucao(v_lar, v_br, RESOLUCAO))
                
    # 2. Extração de Candidatos para Treino (Fundos e Cones)
    for cand in extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO):
        x, y, w, h = cand
        c_lar, c_br = mask_lar[max(0,y):y+h, max(0,x):x+w], mask_br[max(0,y):y+h, max(0,x):x+w]
        if c_lar.size == 0: continue
        
        iou = max([calcular_iou(cand, gab) for gab in gabaritos], default=0.0)
        
        if iou <= ESTADO['iou_negativo']:
            c_lar, c_br = alinhar_cone_vertical(c_lar, c_br, limiar_ar=LIMIAR_AR_CONE) # <-- NOVO
            fundos_X.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO))
        elif iou >= ESTADO['iou_positivo']:
            c_lar, c_br = alinhar_cone_vertical(c_lar, c_br, limiar_ar=LIMIAR_AR_CONE) # <-- NOVO
            cones_X.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO))

random.seed(42)
random.shuffle(fundos_X)
fundos_X = fundos_X[:len(cones_X)]
modelo = wp.Wisard(TUPLA, ignoreZero=IGN_ZERO)
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
    candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO)
    
    # ==========================================================
    # INJEÇÃO DE MEMÓRIA (TRACKING)
    # ==========================================================
    for caixas_antigas in memoria_caixas:
        for box_antiga in caixas_antigas:
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
            if c_lar.size > 0:
                c_lar, c_br = alinhar_cone_vertical(c_lar, c_br, limiar_ar=LIMIAR_AR_CONE) # <-- NOVO
                recortes.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO))
                candidatos_validos.append((x, y, w, h))
                
        if recortes:
            preds = modelo.classify(wp.DataSet(recortes))
            aprovadas_raw = [candidatos_validos[i] for i, p in enumerate(preds) if p == 'cone']
            
            # Filtro NMS Corrigido para Caixas Aninhadas
            for box_raw in sorted(aprovadas_raw, key=lambda b: b[2]*b[3], reverse=True):
                if not any(calcular_iom(box_raw, b_apr) > 0.6 for b_apr in caixas_filtradas):
                    caixas_filtradas.append(box_raw)
                    
            if EXIBIR_TODAS_CAIXAS:
                for idx, box in enumerate(candidatos_validos):
                    cor = (0, 255, 0) if preds[idx] == 'cone' else (0, 0, 255)
                    cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), cor, 1 if preds[idx] != 'cone' else 2)

    # ==========================================================
    # ATUALIZAÇÃO DA MEMÓRIA
    # ==========================================================
    memoria_caixas.append(caixas_filtradas)
    # ==========================================================

    if not EXIBIR_TODAS_CAIXAS:
        for box in caixas_filtradas:
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Cone", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    t_fim = time.perf_counter()
    fps_medio = (fps_medio * 0.9) + ((1.0 / (t_fim - t_inicio)) * 0.1)
    cv2.putText(frame, f"FPS: {fps_medio:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow('WiSARD - Real Time Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()