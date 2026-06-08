import cv2
import numpy as np
import os
import glob
import time
import random
import gc
import json
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
LIMIAR_CONFIANCA = 0.0

# --- Execuções Múltiplas para Cálculo de Incerteza ---
NUM_EXECUCOES = 100
# ==========================================================

if __name__ == "__main__":
    print("="*60)
    print(f" BENCHMARK: PIPELINE WISARD ({NUM_EXECUCOES} EXECUÇÕES)")
    print("="*60)
    print(f" -> Modo Ativo: {MODO_BINARIZACAO.upper()}")
    
    pasta_saida = "resultados_wisard"
    pasta_mental = os.path.join(pasta_saida, "mental_images")
    pasta_validacao = os.path.join(pasta_saida, "test_images")
    
    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(pasta_mental, exist_ok=True)
    os.makedirs(pasta_validacao, exist_ok=True)
    
    print("\n[Fase 1] Carregando e Extraindo Dados de Treino Base...")
    cones_X_base, fundos_X_base = [], []
    for arq in glob.glob("images/train/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        gabaritos = ler_gabarito_yolo(arq, w_img, h_img)
        
        mask_lar, mask_br = gerar_mascaras(img, ESTADO)
        mask_canny = gerar_canny(img, ESTADO) 
        
        for x, y, w, h in gabaritos:
            x, y = max(0, x), max(0, y)
            c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
            c_canny = mask_canny[y:y+h, x:x+w] 
            
            if c_lar.size > 0:
                c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                for v_lar, v_br, v_canny in augmentar_recorte_mascaras(c_lar, c_br, c_canny):
                    cones_X_base.append(binarizar_para_resolucao(v_lar, v_br, RESOLUCAO, v_canny, MODO_BINARIZACAO))
        
        for cand in extrair_candidatos_multiplos(mask_lar, mask_br, w_img, h_img, ESTADO):
            x, y, w, h = cand
            x, y = max(0, x), max(0, y)
            c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
            c_canny = mask_canny[y:y+h, x:x+w] 
            
            if c_lar.size == 0: continue
            iou = max([calcular_iou(cand, gab) for gab in gabaritos], default=0.0)
            
            if iou <= ESTADO['iou_negativo']:
                c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                fundos_X_base.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO, c_canny, MODO_BINARIZACAO))
            elif iou >= ESTADO['iou_positivo']:
                c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                cones_X_base.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO, c_canny, MODO_BINARIZACAO))

    print(f" -> {len(cones_X_base)} recortes de cone extraídos.")
    print(f" -> {len(fundos_X_base)} recortes de fundo (candidatos) extraídos.")

    print("\n[Fase 2] Pré-processando Dados de Teste em Cache...")
    cache_teste = []
    tot_gab_global = 0
    todos_gabaritos_map = {}
    
    for arq in glob.glob("images/test/*.*"):
        img = cv2.imread(arq)
        if img is None: continue
        img_id = os.path.basename(arq)
        gabaritos = ler_gabarito_yolo(arq, img.shape[1], img.shape[0])
        
        todos_gabaritos_map[img_id] = gabaritos
        tot_gab_global += len(gabaritos)
        
        t0_pre = time.perf_counter()
        mask_lar, mask_br = gerar_mascaras(img, ESTADO)
        mask_canny = gerar_canny(img, ESTADO) 
        candidatos = extrair_candidatos_multiplos(mask_lar, mask_br, img.shape[1], img.shape[0], ESTADO)
        
        recortes, candidatos_validos = [], []
        if candidatos:
            for (x, y, w, h) in candidatos:
                x, y = max(0, x), max(0, y)
                c_lar, c_br = mask_lar[y:y+h, x:x+w], mask_br[y:y+h, x:x+w]
                c_canny = mask_canny[y:y+h, x:x+w] 
                
                if c_lar.size > 0:
                    c_lar, c_br, c_canny = alinhar_cone_vertical(c_lar, c_br, c_canny, limiar_ar=LIMIAR_AR_CONE)
                    recortes.append(binarizar_para_resolucao(c_lar, c_br, RESOLUCAO, c_canny, MODO_BINARIZACAO))
                    candidatos_validos.append((x, y, w, h))
                    
        tempo_pre = time.perf_counter() - t0_pre
        
        cache_teste.append({
            'img_id': img_id,
            'img_path': arq,
            'gabaritos': gabaritos,
            'recortes': recortes,
            'candidatos_validos': candidatos_validos,
            'tempo_pre': tempo_pre
        })

    # =========================================================================
    # FASE 3: MÚLTIPLAS EXECUÇÕES (Treino + Inferência)
    # =========================================================================
    print(f"\n[Fase 3] Iniciando o ciclo de {NUM_EXECUCOES} rodadas...")
    
    resultados = {
        'map50': [], 'map50_95': [], 'precisao': [], 
        'recall': [], 'f1': [], 'iou': [], 'tempo_inferencia': [], 'tempo_treino': []
    }
    
    melhor_f1_global = -1.0
    melhores_caixas_por_imagem = {}
    metricas_melhor_modelo = {}

    for rodada in range(NUM_EXECUCOES):
        print(f"\n--- RODADA {rodada+1}/{NUM_EXECUCOES} ---")
        
        # 1. Amostragem Aleatória de Fundos (Under-sampling)
        random.seed(42 + rodada)
        fundos_X_rodada = fundos_X_base.copy()
        random.shuffle(fundos_X_rodada)
        fundos_X_rodada = fundos_X_rodada[:len(cones_X_base)]
        
        X_train = cones_X_base + fundos_X_rodada
        y_train = ['cone'] * len(cones_X_base) + ['nao_cone'] * len(fundos_X_rodada)
        
        # Embaralhamento das amostras de treino
        combinado = list(zip(X_train, y_train))
        random.shuffle(combinado)
        X_train, y_train = zip(*combinado)
        X_train, y_train = list(X_train), list(y_train)
        
        # 2. Treinamento
        t0_train = time.perf_counter()
        modelo = wp.Wisard(TUPLA, ignoreZero=IGN_ZERO, returnConfidence=True)
        modelo.train(wp.DataSet(X_train, y_train))
        tempo_treino_rodada = time.perf_counter() - t0_train
                    
        # 3. Inferência (Usando o Cache)
        tempos_rodada, ious_acertos_rodada = [], []
        tp_ofc, tp_rel, fp = 0, 0, 0
        todas_predicoes_map = []
        caixas_da_rodada = {}
        
        for img_data in cache_teste:
            img_id = img_data['img_id']
            gabaritos = img_data['gabaritos']
            caixas_filtradas = []
            
            if img_data['recortes']:
                dataset_test = wp.DataSet(img_data['recortes'])
                
                # Classificação
                t0_class = time.perf_counter()
                votos_preds = modelo.rank(dataset_test)
                tempo_class = time.perf_counter() - t0_class
                
                tempos_rodada.append((img_data['tempo_pre'] + tempo_class) * 1000)

                candidatos_com_score = []
                for i, votos_dict in enumerate(votos_preds):
                    classe_predita = max(votos_dict, key=votos_dict.get)
                    votos_vencedor = votos_dict[classe_predita]
                    total_votos = sum(votos_dict.values())
                    confianca = (votos_vencedor / total_votos) if total_votos > 0 else 0.0

                    if classe_predita == 'cone' and confianca >= LIMIAR_CONFIANCA:
                        candidatos_com_score.append({
                            'box': img_data['candidatos_validos'][i],
                            'score': confianca,
                            'votos_dict': votos_dict
                        })

                candidatos_com_score.sort(key=lambda item: item['box'][2] * item['box'][3], reverse=True)
                
                for item in candidatos_com_score:
                    box_raw = item['box']
                    if not any(calcular_iom(box_raw, b_apr['box']) > 0.6 for b_apr in caixas_filtradas):
                        caixas_filtradas.append(item)
                        todas_predicoes_map.append({
                            'img_id': img_id,
                            'box': item['box'],
                            'score': item['score']
                        })

                # Armazena em RAM as caixas dessa rodada
                caixas_da_rodada[img_id] = {
                    'img_path': img_data['img_path'],
                    'candidatos_validos': img_data['candidatos_validos'],
                    'caixas_filtradas': caixas_filtradas.copy(),
                    'votos_preds': votos_preds if EXIBIR_TODAS_CAIXAS else []
                }
            else:
                tempos_rodada.append(img_data['tempo_pre'] * 1000)
                caixas_da_rodada[img_id] = {
                    'img_path': img_data['img_path'],
                    'candidatos_validos': [],
                    'caixas_filtradas': [],
                    'votos_preds': []
                }

            # Cálculo TP, FP para P/R/F1 normal
            gab_det_ofc, gab_det_rel = set(), set()
            for item in caixas_filtradas:
                box = item['box']
                melhor_iou, idx_gab = max([(calcular_iou(box, g), i) for i, g in enumerate(gabaritos)], default=(0.0, -1))
                is_tp = False
                if melhor_iou >= 0.50:
                    if idx_gab not in gab_det_ofc:
                        tp_ofc += 1
                        gab_det_ofc.add(idx_gab)
                        ious_acertos_rodada.append(melhor_iou)
                    is_tp = True
                if melhor_iou >= 0.30 and idx_gab not in gab_det_rel:
                    tp_rel += 1
                    gab_det_rel.add(idx_gab)
                if not is_tp: fp += 1

        precisao_r = (tp_ofc / (tp_ofc + fp)) * 100 if (tp_ofc + fp) > 0 else 0
        recall_r = (tp_ofc / tot_gab_global) * 100 if tot_gab_global > 0 else 0
        f1_r = 2 * (precisao_r * recall_r) / (precisao_r + recall_r) if (precisao_r + recall_r) > 0 else 0
        map50_r, map50_95_r = calcular_map_coco(todas_predicoes_map, todos_gabaritos_map)
        iou_medio_r = np.mean(ious_acertos_rodada) * 100 if ious_acertos_rodada else 0.0
        tempo_medio_r = np.mean(tempos_rodada)

        print(f" -> Treino: {tempo_treino_rodada:.2f}s | mAP@50: {map50_r*100:.2f}% | F1: {f1_r:.2f}%")

        # =====================================================================
        # Salva modelos, imagens mentais e predições apenas se for o melhor
        # =====================================================================
        if f1_r > melhor_f1_global:
            melhor_f1_global = f1_r
            print(f"    [!] Novo melhor modelo (F1: {f1_r:.2f}%). Guardando configurações...")
            
            metricas_melhor_modelo = {
                'map50': map50_r * 100,
                'map50_95': map50_95_r * 100,
                'precisao': precisao_r,
                'recall': recall_r,
                'f1': f1_r,
                'iou': iou_medio_r,
                'tempo_treino': tempo_treino_rodada,
                'tempo_inferencia': tempo_medio_r
            }
            
            melhores_caixas_por_imagem = caixas_da_rodada.copy()

            caminho_modelo = os.path.join(pasta_saida, "melhor_modelo.json")
            with open(caminho_modelo, 'w') as f:
                f.write(modelo.json())
            
            # Limpa imagens antigas antes de salvar as novas
            for f in glob.glob(os.path.join(pasta_mental, "*.png")):
                os.remove(f)
                
            patterns = modelo.getMentalImages()
            for classe, padrao_lista in patterns.items():
                padrao = np.array(padrao_lista, dtype=np.float32)
                tamanho_metade = RESOLUCAO * RESOLUCAO
                imagens_concatenadas = []
                
                if MODO_BINARIZACAO in ['cor', 'hibrido']:
                    if len(padrao) >= 2 * tamanho_metade:
                        mental_laranja = padrao[:tamanho_metade].reshape((RESOLUCAO, RESOLUCAO))
                        mental_branco = padrao[tamanho_metade:2*tamanho_metade].reshape((RESOLUCAO, RESOLUCAO))
                        imagens_concatenadas.extend([normalizar_para_imagem(mental_laranja), normalizar_para_imagem(mental_branco)])
                    
                if MODO_BINARIZACAO in ['canny', 'hibrido']:
                    offset = 2 * tamanho_metade if MODO_BINARIZACAO == 'hibrido' else 0
                    if len(padrao) >= offset + tamanho_metade:
                        mental_canny = padrao[offset:offset+tamanho_metade].reshape((RESOLUCAO, RESOLUCAO))
                        imagens_concatenadas.append(normalizar_para_imagem(mental_canny))
                
                if imagens_concatenadas:
                    imagem_final = cv2.hconcat(imagens_concatenadas)
                    cv2.imwrite(os.path.join(pasta_mental, f"mental_image_{classe}_{MODO_BINARIZACAO}.png"), imagem_final)
        # =====================================================================

        resultados['map50'].append(map50_r * 100)
        resultados['map50_95'].append(map50_95_r * 100)
        resultados['precisao'].append(precisao_r)
        resultados['recall'].append(recall_r)
        resultados['f1'].append(f1_r)
        resultados['iou'].append(iou_medio_r)
        resultados['tempo_inferencia'].append(tempo_medio_r)
        resultados['tempo_treino'].append(tempo_treino_rodada)
        
        # Limpar memória RAM do modelo antigo
        del modelo
        gc.collect()

    # =========================================================================
    # FASE 4: EXPORTAR IMAGENS DE VALIDAÇÃO DO MELHOR MODELO
    # =========================================================================
    print(f"\n[Fase 4] Salvando imagens de validação do melhor modelo encontrado...")
    
    # Limpa imagens antigas de validação (se houver)
    for f in glob.glob(os.path.join(pasta_validacao, "*.png")):
        os.remove(f)
    for f in glob.glob(os.path.join(pasta_validacao, "*.jpg")):
        os.remove(f)

    for img_id, dados in melhores_caixas_por_imagem.items():
        img = cv2.imread(dados['img_path'])
        if img is None: continue

        if EXIBIR_TODAS_CAIXAS:
            for idx, box in enumerate(dados['candidatos_validos']):
                if idx < len(dados['votos_preds']):
                    votos_dict = dados['votos_preds'][idx]
                    classe_box_raw = max(votos_dict, key=votos_dict.get)
                    classe_box = classe_box_raw.split('::')[0]
                    cor = (0, 255, 0) if classe_box == 'cone' else (0, 0, 255)
                    espessura = 2 if classe_box == 'cone' else 1
                    cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), cor, espessura)
        else:
            for item in dados['caixas_filtradas']:
                x, y, w, h = item['box']
                score = item['score']
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"Cone {score:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(pasta_validacao, img_id), img)

   # =========================================================================
    # RELATÓRIO FINAL E EXPORTAÇÃO JSON
    # =========================================================================
    print("\n" + "="*60)
    print(f" RESULTADO FINAL ({NUM_EXECUCOES} RODADAS)")
    print("="*60)
    print(" [MÉDIA ± INCERTEZA]")
    print(f" -> mAP@50         : {np.mean(resultados['map50']):.2f}% ± {np.std(resultados['map50']):.2f}%")
    print(f" -> mAP@50-95      : {np.mean(resultados['map50_95']):.2f}% ± {np.std(resultados['map50_95']):.2f}%")
    print(f" -> Precisão       : {np.mean(resultados['precisao']):.2f}% ± {np.std(resultados['precisao']):.2f}%")
    print(f" -> Recall         : {np.mean(resultados['recall']):.2f}% ± {np.std(resultados['recall']):.2f}%")
    print(f" -> F1-Score       : {np.mean(resultados['f1']):.2f}% ± {np.std(resultados['f1']):.2f}%")
    print(f" -> Média de IoU   : {np.mean(resultados['iou']):.2f}% ± {np.std(resultados['iou']):.2f}%")
    print(f" -> T. Treino/Ciclo: {np.mean(resultados['tempo_treino']):.4f}s ± {np.std(resultados['tempo_treino']):.4f}s")
    print(f" -> T. Inf./Imagem : {np.mean(resultados['tempo_inferencia']):.2f}ms ± {np.std(resultados['tempo_inferencia']):.2f}ms")
    print("-" * 60)
    print(" [MÉTRICAS DO MELHOR MODELO (Top F1)]")
    if metricas_melhor_modelo:
        print(f" -> mAP@50         : {metricas_melhor_modelo['map50']:.2f}%")
        print(f" -> mAP@50-95      : {metricas_melhor_modelo['map50_95']:.2f}%")
        print(f" -> Precisão       : {metricas_melhor_modelo['precisao']:.2f}%")
        print(f" -> Recall         : {metricas_melhor_modelo['recall']:.2f}%")
        print(f" -> F1-Score       : {metricas_melhor_modelo['f1']:.2f}%")
        print(f" -> Média de IoU   : {metricas_melhor_modelo['iou']:.2f}%")
        print(f" -> Tempo Treino   : {metricas_melhor_modelo['tempo_treino']:.4f}s")
        print(f" -> Tempo Inferencia: {metricas_melhor_modelo['tempo_inferencia']:.2f}ms")
    print("="*60)

    # Estruturando e Salvando JSON
    relatorio_json = {
        "configuracoes": {
            "num_execucoes": NUM_EXECUCOES,
            "modo_binarizacao": MODO_BINARIZACAO,
            "resolucao": RESOLUCAO,
            "tupla": TUPLA,
            "ignore_zero": IGN_ZERO,
            "limiar_ar_cone": LIMIAR_AR_CONE,
            "limiar_confianca": LIMIAR_CONFIANCA
        },
        "melhor_modelo": {k: float(v) for k, v in metricas_melhor_modelo.items()},
        "sumario_medias": {
            "map50": {"media": float(np.mean(resultados['map50'])), "incerteza": float(np.std(resultados['map50']))},
            "map50_95": {"media": float(np.mean(resultados['map50_95'])), "incerteza": float(np.std(resultados['map50_95']))},
            "precisao": {"media": float(np.mean(resultados['precisao'])), "incerteza": float(np.std(resultados['precisao']))},
            "recall": {"media": float(np.mean(resultados['recall'])), "incerteza": float(np.std(resultados['recall']))},
            "f1_score": {"media": float(np.mean(resultados['f1'])), "incerteza": float(np.std(resultados['f1']))},
            "iou_medio": {"media": float(np.mean(resultados['iou'])), "incerteza": float(np.std(resultados['iou']))},
            "tempo_treino_s": {"media": float(np.mean(resultados['tempo_treino'])), "incerteza": float(np.std(resultados['tempo_treino']))},
            "tempo_inferencia_ms": {"media": float(np.mean(resultados['tempo_inferencia'])), "incerteza": float(np.std(resultados['tempo_inferencia']))}
        },
        "historico_rodadas": {
            "map50": [float(x) for x in resultados['map50']],
            "map50_95": [float(x) for x in resultados['map50_95']],
            "precisao": [float(x) for x in resultados['precisao']],
            "recall": [float(x) for x in resultados['recall']],
            "f1": [float(x) for x in resultados['f1']],
            "iou": [float(x) for x in resultados['iou']],
            "tempo_treino": [float(x) for x in resultados['tempo_treino']],
            "tempo_inferencia": [float(x) for x in resultados['tempo_inferencia']]
        }
    }
    
    nome_arquivo_json = os.path.join(pasta_saida, "relatorio_benchmark_wisard.json")
    with open(nome_arquivo_json, 'w', encoding='utf-8') as f:
        json.dump(relatorio_json, f, indent=4, ensure_ascii=False)
        
    print(f"\n[*] Resultados exportados com sucesso para: {nome_arquivo_json}")