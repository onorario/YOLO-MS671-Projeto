# 👁️ YOLOv8-nano: Reconhecimento de Objetos via YOLOv8 - Projeto 01 (MS761)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)

> **Contexto:** Projeto desenvolvido para a disciplina **MS671 - Introdução ao Aprendizado de Máquina Profundo** (IMECC / UNICAMP).

Foi realizado a implementação da arquitetura **YOLOv8-nano** desenvolvida na linguagem **Python**, carregando apenas os tensores brutos dos pesos oficiais (`yolov8n.pt`) e reconstruindo manualmente todo o pipeline de inferência, decodificação de caixas (*Distribution Focal Loss - DFL*) e algoritmos de pós-processamento (**IoU** e **Non-Maximum Suppression - NMS**).



---

## 📌 Destaques do Projeto

- Implementação manualmente do pós-processamento das detecções.
- **Mapeamento Explícito de Pesos:** Algoritmo customizado para inspecionar, mapear e carregar `state_dict` oficial nas camadas manuais.
- **Pós-processamento Manual:**
  - Cálculo matricial de **IoU** (`manual_iou`).
  - Algoritmo de **Supressão Não-Máxima** (`manual_nms`) aplicado por classe.
- **Estudo de Sensibilidade e Hiperparâmetros:** Avaliação empírica do impacto dos limiares de confiança ($0.05 \dots 0.75$) e de IoU ($0.10 \dots 0.90$) em cenários simples e de alta densidade visual.

---

## 🧠 Como o Modelo "Enxerga"? (Visão Geral da Arquitetura)

O projeto foi desenvolvido em Python utilizando o modelo YOLOv8n da biblioteca Ultralytics.

### 1. 👁️ Backbone (A Extração de Características)
É como se fossem os "olhos" da rede neural. Conforme a imagem percorre as camadas convolucionais, o modelo vai aprendendo padrões em diferentes níveis:
* **Primeiras camadas:** Identificam detalhes simples, como bordas, cores e texturas.
* **Camadas mais profundas:** Combinam esses detalhes para reconhecer formatos complexos

---

### 2. 🧠 Neck (A Combinação de Escalas)
Aqui é avaliado um dos maiores desafios em visão computacional, que é conseguir detectar **tanto objetos minúsculos quanto objetos enormes** na mesma cena:
* O módulo *Neck* mistura informações vindas de diferentes profundidades da rede.
* Ele garante que o modelo não perca a noção do contexto geral da foto enquanto ainda presta atenção em pequenos detalhes ao fundo.

---

### 3. 🎯 Head (A Decisão Final e os Cálculos)
Saída do modelo, onde as características visuais são convertidas em previsões numéricas diretas:
* **O que é o objeto? (Classificação):** Calcula a probabilidade de pertencer a uma das 80 classes do dataset COCO.
* **Onde ele está? (Localização):** Em vez de "chutar" uma caixa pronta, o modelo estima matematicamente o centro e as distâncias até as bordas (*Distribution Focal Loss*), desenhando o retângulo final (*bounding box*).

---

### ⚙️ E depois do modelo? (O Pós-Processamento Manual)
Aplicações das funções criadas, local onde se ajusta para definir se aquele é um objetivo válido ou não:
1. **Filtro de Confiança:** Descarta qualquer palpite que tenha certeza matemática baixa.
2. **IoU (*Intersection over Union*):** Mede a porcentagem de sobreposição geométrica entre duas caixas.
3. **NMS (*Non-Maximum Suppression*):** Algoritmo que compara as caixas sobrepostas, mantém apenas a de maior confiança e elimina as cópias redundantes.

---

## 📊 Análise e Resultados Experimentais

A robustez da detecção e a sensibilidade do algoritmo foram testadas sob 4 valores principais:

| Cenário | Confiança ($\tau_{score}$) | IoU NMS ($\tau_{IoU}$) | Comportamento Observado |
| :--- | :---: | :---: | :--- |
| **Baixa Confiança / IoU Baixo** | $0.10$ | $0.30$ | Alto recall, porém com aparecimento de falsos positivos em texturas densas. |
| **Padrão YOLOv8** | $0.25$ | $0.45$ | Melhor equilíbrio entre precisão e recall na maioria das cenas. |
| **Alta Confiança / IoU Alto** | $0.50$ | $0.60$ | Filtragem rígida; risco de duplicação de caixas para objetos muito próximos. |
| **Confiança Muito Alta** | $0.70$ | $0.45$ | Alta precisão, mas com falsos negativos severos para objetos distantes/ocultos. |

### Principais Insights
- **Ruído Visual vs. Estabilidade:** Imagens com objeto único e alto contraste (e.g. `cat.jpg`) mantêm **100% de estabilidade** (1 detecção) independentemente do limiar.
- **Cenas Densas (e.g. Avenida Paulista):** O limiar de IoU exerce papel crítico. Valores de IoU $> 0.70$ causam acúmulo de *bounding boxes* sobrepostas em multidões.

---


