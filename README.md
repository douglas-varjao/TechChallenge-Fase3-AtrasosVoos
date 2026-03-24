# ✈️ TechChallenge - Fase3: Atrasos de Voos

**Dataset:** ~5,7 milhões de voos domésticos nos EUA em 2015 - [Acessar Base de Dados](https://drive.google.com/drive/u/0/folders/1fm1c9CqkzhSaC4RE6DZNQwOzm1KAOOqC)  
**Problema:** Classificação binária — o voo vai atrasar mais de 15 minutos?  
**Critério:** FAA (Federal Aviation Administration) — atraso oficial > 15 min na chegada  
**Destaques Técnicos:** K-Means, PCA, LightGBM, XGBoost, MLP e SHAP Values.

---

## 📖 Índice

1. [Decisões de Limpeza](#1-decisões-de-limpeza) 
2. [Feature Engineering e Dados Externos](#2-feature-engineering-e-dados-externos)
3. [Inteligência Não Supervisionada (K-Means e PCA)](#3-inteligência-não-supervisionada-k-means-e-pca)
4. [Modelagem Supervisionada e Desbalanceamento](#4-modelagem-supervisionada-e-desbalanceamento)
5. [Resultados e Explicabilidade (SHAP)](#5-resultados-e-explicabilidade-shap)
6. [Análise de Erros — Falsos Negativos](#6-análise-de-erros--falsos-negativos)
7. [Limitações e Próximos Passos](#7-limitações-e-próximos-passos)

---

## 🚀 Como Executar o Projeto (Acesso aos Dados)

Como os arquivos originais em formato CSV (`flights.csv`, `airlines.csv`, `airports.csv` e `clima_2015.csv`) excedem o limite de tamanho permitido pelo GitHub (100MB), os dados foram alocados em nuvem. 

Para reproduzir o código no Google Colab, siga os passos:

1. Acesse a base de dados através deste **[Link do Google Drive](https://drive.google.com/drive/u/0/folders/1fm1c9CqkzhSaC4RE6DZNQwOzm1KAOOqC)**.
2. Faça o download dos arquivos ou crie um atalho para o seu próprio Drive.
3. No Passo 00/01 do *notebook*, o código fará a montagem do Drive (`drive.mount`). Certifique-se de que a variável `PATH` reflete a estrutura de pastas do seu ambiente. O padrão configurado no código é:

```python
PATH = 'drive/MyDrive/TechChallenge_Fase3_AtrasosVoos/database/'
```
---

## 1. Decisões de Limpeza

| Decisão | Justificativa de Negócio |
|---|---|
| **Remove Cancelados/Desviados** | Eventos de natureza diferente; misturá-los polui a decisão do classificador. |
| **Remove sem `ARRIVAL_DELAY`** | Sem a variável alvo (Target), não há aprendizado supervisionado. |
| **Remove Outliers (> 600 min)** | Atrasos extremos (>10h) têm causas únicas (ex: furacões) e distorcem a distribuição normal. |

## 2. Feature Engineering e Dados Externos

Para captar a complexidade da malha aérea *antes* da decolagem, criamos variáveis baseadas em conhecimento de domínio:

* **Histórico de Cascata:** Média móvel (`rolling(7)`) de atrasos da mesma rota, capturando problemas contínuos.
* **Calendário:** *Flags* para feriados federais (`IS_HOLIDAY`), vésperas e finais de semana.
* **Fluxo Diário:** Contagem total de voos no aeroporto de origem no dia (mensura congestionamento de *hubs*).
* **Clima (API Externa):** Integração com a API Open-Meteo adicionando dados reais de 2015 (`chuva_mm`, `vento_kmh` e `temperatura_max`).

## 3. Inteligência Não Supervisionada (K-Means e PCA)

Antes de prever atrasos individuais, agrupamos os aeroportos por perfil de risco operacional:
* **Clusterização (K-Means):** Usamos métricas como % de causas de atraso (clima, falha de companhia, cascata) para gerar 4 *clusters* distintos (ex: Eficientes, Críticos, Efeito Cascata e Sensíveis ao Clima).
* **Feature Injetada:** O resultado do K-Means virou a variável `CLUSTER_AERO`, alimentando os modelos preditivos.
* **Redução de Dimensionalidade:** Usamos PCA em 2D para validar visualmente a separação matemática dos clusters.

## 4. Modelagem Supervisionada e Desbalanceamento

Como apenas ~18% dos voos atrasam, o dataset é fortemente desbalanceado. 
* **Tratamento:** Uso do `stratify` no *split* de dados e hiperparâmetro `scale_pos_weight` penalizando erros na classe minoritária.
* **Modelos Testados:**
    1.  **XGBoost:** *Baseline* forte para tabelas (*level-wise growth*).
    2.  **LightGBM:** Mais veloz e eficiente na ramificação (*leaf-wise growth*).
    3.  **Rede Neural (MLP):** Camadas `(128, 64, 32)` com `StandardScaler` para capturar relações não-lineares severas.
* **Threshold Ajustado:** Reduzimos o ponto de corte para **0.4**, sacrificando levemente a *Precision* em favor do *Recall* (redução drástica de Falsos Negativos).

## 5. Resultados e Explicabilidade (SHAP)

O **LightGBM** obteve o melhor balanço de F1-Score e velocidade de treinamento. Para evitar o efeito "caixa-preta", aplicamos técnicas de XAI:
* **Feature Importance Global:** `HIST_ATRASO_ROTA` e `HORA` dominaram os ganhos da árvore.
* **SHAP Values (Beeswarm):** Comprovou que voos no fim do dia (alta `HORA`) empurram forte e positivamente a probabilidade de atraso.
* **SHAP Dependence:** Mostrou a não-linearidade do impacto do horário em relação às faixas de risco.

## 6. Análise de Erros — Falsos Negativos

Usando o *SHAP Waterfall Plot*, dissecamos os erros do modelo (voos que atrasaram sem o modelo prever):
* **Diagnóstico Sistêmico:** Companhias como a Delta Airlines concentram falsos negativos no aeroporto de Atlanta (ATL).
* **Conclusão:** A *feature* de cascata atual (baseada na rota) é "cega" para engarrafamentos sistêmicos em *mega-hubs*. O modelo previu uma probabilidade hesitante (~0.35) em muitos desses casos.

## 7. Limitações e Próximos Passos

1.  **Clima Horário:** Atualmente o clima é diário; migrar para previsão por hora capturaria tempestades de fim de tarde.
2.  **Clusterização Dinâmica:** Refazer o agrupamento K-Means por estação do ano (inverno vs. verão).
3.  **Features Hiper-específicas:** Criar histórico cruzando `[Companhia + Aeroporto + Mês]` para sanar o erro descoberto em hubs como o da Delta.
