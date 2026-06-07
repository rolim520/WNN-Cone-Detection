import cv2
import numpy as np
import os
import glob
import time
import random
import wisardpkg as wp
from utils import *

ESTADO = carregar_configuracoes()

# ==========================================================
# PARÂMETROS OTIMIZADOS (Refletindo o Top 1 do Grid Search)
# ==========================================================
MODO_BINARIZACAO = 'cor' # Opções: 'cor', 'canny', 'hibrido'
RESOLUCAO = 64
TUPLA = 16
IGN_ZERO = False

EXIBIR_TODAS_CAIXAS = False
LIMIAR_AR_CONE = 1.25
# ==========================================================

if __name__ == "__main__":
    print("="*60)
    print(" BENCHMARK FINAL: PIPELINE SEM PESOS vs TRACON (YOLOv5)")
    print("="*60)
    print(f" -> Modo Ativo: {MODO_BINARIZACAO.upper()}")
    
    print("\n[Fase 1] Carregando Dados de Treino...")
    cones_X, fundos_X = [], []
    for arq in glob.glob("images/train/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
        
        mask_lar, mask_br = gerar_mascaras(img, ESTADO)
        mask_canny = gerar_canny(img, ESTADO) # <- Canny gerado
        
        for x, y, w, h in gabaritos:
            x, y = max(0, x), max(0, y)
            c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
            c_canny = mask_canny[y:y+h, x:x+w] # <- Recorte Canny
            
            if c_lar.size > 0:
                c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                for v_lar, v_br, v_canny in augmentar_recorte_mascaras(c_lar, c_br, c_canny):
                    cones_X.append(binarizar_para_resolucao(v_lar, v_br, RESOLUCAO, v_canny, MODO_BINARIZACAO))
        
        for cand in extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO):
            x, y, w, h = cand
            x, y = max(0, x), max(0, y)
            c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
            c_canny = mask_canny[y:y+h, x:x+w] # <- Recorte Canny
            
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
    X_train = cones_X + fundos_X
    y_train = ['cone'] * len(cones_X) + ['nao_cone'] * len(fundos_X)
    
    print(f"\n[Fase 2] Treinando WiSARD (Res={RESOLUCAO}, Tupla={TUPLA})...")
    modelo = wp.Wisard(TUPLA, ignoreZero=IGN_ZERO)
    modelo.train(wp.DataSet(X_train, y_train))

    # Atualizado para receber o modo de binarização
    salvar_imagem_mental(modelo, resolucao=RESOLUCAO, modo=MODO_BINARIZACAO)

    print("\n[Fase 3] Rodando Inferência no Conjunto de Teste End-to-End...")
    pasta_saida = "resultados_finais"
    os.makedirs(pasta_saida, exist_ok=True)
    
    tempos, ious_acertos = [], []
    tp_ofc, tp_rel, fp, tot_gab = 0, 0, 0, 0
    
    for arq in glob.glob("images/test/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        gabaritos = ler_gabarito_yolo(arq, img.shape[1], img.shape[0])
        tot_gab += len(gabaritos)
        
        t0 = time.perf_counter()
        mask_lar, mask_br = gerar_mascaras(img, ESTADO)
        mask_canny = gerar_canny(img, ESTADO) # <- Canny gerado
        
        candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, img.shape[1], img.shape[0], ESTADO)
        caixas_filtradas = []
        
        if candidatos:
            recortes, candidatos_validos = [], []
            for (x, y, w, h) in candidatos:
                x, y = max(0, x), max(0, y)
                c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
                c_canny = mask_canny[y:y+h, x:x+w] # <- Recorte Canny
                
                if c_lar.size > 0:
                    c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                    recortes.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO, c_canny, MODO_BINARIZACAO))
                    candidatos_validos.append((x, y, w, h))
                    
            if recortes:
                preds = modelo.classify(wp.DataSet(recortes))
                aprovadas_raw = [candidatos_validos[i] for i, p in enumerate(preds) if p == 'cone']
                # Filtro NMS Corrigido para Caixas Aninhadas
                for box_raw in sorted(aprovadas_raw, key=lambda b: b[2]*b[3], reverse=True):
                    # Usando IoM! Se uma caixa estiver 60% ou mais engolida por outra, é deletada.
                    if not any(calcular_iom(box_raw, b_apr) > 0.6 for b_apr in caixas_filtradas):
                        caixas_filtradas.append(box_raw)

                # --- EXIBIR TODAS AS CAIXAS ---
                if EXIBIR_TODAS_CAIXAS:
                    for idx, box in enumerate(candidatos_validos):
                        cor = (0, 255, 0) if preds[idx] == 'cone' else (0, 0, 255)
                        cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), cor, 1 if preds[idx] != 'cone' else 2)

        tempos.append((time.perf_counter() - t0) * 1000)
        
        gab_det_ofc, gab_det_rel = set(), set()
        for box in caixas_filtradas:
            melhor_iou, idx_gab = max([(calcular_iou(box, g), i) for i, g in enumerate(gabaritos)], default=(0.0, -1))
            is_tp = False
            if melhor_iou >= 0.50:
                if idx_gab not in gab_det_ofc:
                    tp_ofc += 1
                    gab_det_ofc.add(idx_gab)
                    ious_acertos.append(melhor_iou)
                is_tp = True
            if melhor_iou >= 0.30 and idx_gab not in gab_det_rel:
                tp_rel += 1
                gab_det_rel.add(idx_gab)
            if not is_tp: fp += 1
                
        # --- EXIBIÇÃO PADRÃO SE A VARIÁVEL FOR FALSE ---
        if not EXIBIR_TODAS_CAIXAS:
            for box in caixas_filtradas:
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Cone", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        cv2.imwrite(os.path.join(pasta_saida, os.path.basename(arq)), img)

    precisao = (tp_ofc / (tp_ofc + fp)) * 100 if (tp_ofc + fp) > 0 else 0
    recall = (tp_ofc / tot_gab) * 100 if tot_gab > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    print("\n" + "="*60)
    print(f" -> Precisão : {precisao:.2f}% | Recall : {recall:.2f}% | F1: {f1:.2f}%")
    print(f" -> Média de IoU nos Acertos: {np.mean(ious_acertos)*100:.2f}%" if ious_acertos else " -> Média de IoU nos Acertos: 0.00%")
    print(f" -> Velocidade: {np.mean(tempos):.2f} ms por imagem")