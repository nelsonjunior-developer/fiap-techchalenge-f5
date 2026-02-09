# Análise das Bases 2022, 2023, 2024 e Dicionário de Dados

## Objetivo
Documentar diferenças entre as bases anuais, limitações do dicionário de dados, principais problemas de qualidade e implicações técnicas para ingestão e pré-processamento antes do pipeline de Machine Learning.

## Fontes analisadas
- `dataset/DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx`
  - Abas: `PEDE2022`, `PEDE2023`, `PEDE2024`
- `dataset/DATATHON/Dicionário Dados Datathon.pdf`
- `dataset/DATATHON/PEDE_ Pontos importantes.docx`
- `dataset/Datathon - Machine Learning Engineering.pdf`

## Resumo executivo
- A ingestão não pode ser direta, porque há divergências de schema entre anos, colunas duplicadas e valores inválidos em campos numéricos.
- A variável de interesse de defasagem aparece como `Defas` (2022) e `Defasagem` (2023/2024), exigindo padronização.
- Há interseção parcial de estudantes entre anos, então a construção do dataset temporal precisa de coorte por `RA`.
- O dicionário é útil conceitualmente, mas não cobre integralmente o layout operacional atual (principalmente 2024 e variações de nomenclatura).

## 1) Comparativo estrutural das bases

| Base | Linhas | Colunas | Coluna de defasagem | Observações de schema |
|---|---:|---:|---|---|
| `PEDE2022` | 861 | 42 | `Defas` | Notas como `Matem`, `Portug`, `Inglês`; idade como `Idade 22` |
| `PEDE2023` | 1014 | 48 | `Defasagem` | Notas como `Mat`, `Por`, `Ing`; presença de `INDE 2023`; `Data de Nasc` |
| `PEDE2024` | 1156 | 50 | `Defasagem` | Notas como `Mat`, `Por`, `Ing`; presença de `INDE 2024`; campos extras (`Escola`, `Ativo/ Inativo`) |

## 2) Diferenças entre 2022, 2023 e 2024

### 2.1 Nomenclatura e campos equivalentes
- Defasagem:
  - 2022: `Defas`
  - 2023/2024: `Defasagem`
- Notas acadêmicas:
  - 2022: `Matem`, `Portug`, `Inglês`
  - 2023/2024: `Mat`, `Por`, `Ing`
- Idade/Data de nascimento:
  - 2022: `Idade 22`
  - 2023/2024: `Idade`, `Data de Nasc`
- INDE:
  - 2022: `INDE 22`
  - 2023: `INDE 2023` e também `INDE 23`
  - 2024: `INDE 2024` e também `INDE 23`

### 2.2 Colunas duplicadas no cabeçalho
- `PEDE2023`: `Destaque IPV` duplicada
- `PEDE2024`: `Ativo/ Inativo` duplicada

## 3) Limitação do dicionário de dados
- O dicionário (`Dicionário Dados Datathon.pdf`) descreve bem conceitos de negócio e métricas, mas não reflete integralmente o schema operacional atual do arquivo XLSX.
- Existem diferenças de nomenclatura e campos não descritos explicitamente na mesma granularidade da base atual.
- Em termos práticos: o dicionário deve ser usado como referência conceitual, não como contrato técnico final de ingestão.

## 4) Principais problemas de qualidade identificados

### 4.1 Valores inválidos em campos numéricos
Foram identificados tokens inválidos típicos de planilha em colunas numéricas:
- `#N/A`
- `#DIV/0!`
- `INCLUIR`

Contagem de ocorrências (estimativa por varredura da planilha):
- `PEDE2022`: 0
- `PEDE2023`: 388
- `PEDE2024`: 761

### 4.2 Formato inconsistente de datas
- `PEDE2023` apresenta mistura de formatos em `Data de Nasc` (datas textuais e serial Excel), exigindo normalização antes do uso.

### 4.3 Mudança de categorias textuais entre anos
- Exemplo observado em gênero e instituição de ensino com rótulos semanticamente equivalentes, porém não idênticos.

## 5) Coorte temporal por RA (evidência quantitativa)

Interseção de estudantes entre anos:
- `PEDE2022 ∩ PEDE2023`: 600 RAs
  - 69,8% de `PEDE2022`
  - 59,2% de `PEDE2023`
- `PEDE2022 ∩ PEDE2024`: 472 RAs
  - 54,9% de `PEDE2022`
  - 40,8% de `PEDE2024`
- `PEDE2023 ∩ PEDE2024`: 765 RAs
  - 75,4% de `PEDE2023`
  - 66,2% de `PEDE2024`

Implicação:
- O recorte temporal do modelo deve usar apenas pares válidos `t -> t+1` com `RA` presente nos dois anos, para manter consistência estatística do target.

## 6) Por que a ingestão não pode ser direta
- Colunas equivalentes mudam de nome entre anos.
- Existem cabeçalhos duplicados.
- Há valores inválidos em campos que deveriam ser numéricos.
- Há inconsistência de formatação em datas.
- A base de estudantes não é idêntica entre anos.

Sem tratamento prévio, o pipeline de treino/inferência fica sujeito a falhas silenciosas e leakage operacional.

## 7) Implicações técnicas para ingestão e pré-processamento

### 7.1 Ingestão
- Padronizar headers e resolver duplicidades.
- Aplicar mapeamento de colunas equivalentes por ano.
- Validar presença de colunas obrigatórias (`RA`, defasagem, notas, indicadores críticos).

### 7.2 Qualidade e tipagem
- Converter tokens inválidos para nulos e registrar auditoria.
- Normalizar datas para formato único.
- Padronizar categorias textuais (ex.: gênero e tipo de instituição).

### 7.3 Recorte temporal
- Construir coorte por `RA` em anos consecutivos.
- Calcular e registrar estatísticas de interseção para rastreabilidade do recorte.

### 7.4 Pré-processamento
- Tratar missing após padronização.
- Garantir que transformações de treino sejam reaplicáveis na inferência.
- Impedir uso de informação futura no conjunto de features.

## 8) Contrato mínimo de ingestão (pass/fail)

### Regras de bloqueio
- Falha se houver coluna obrigatória ausente.
- Falha se houver header duplicado não resolvido.
- Falha se colunas críticas tiverem tipo incompatível após padronização.

### Regras de alerta
- Alerta para aumento incomum de valores inválidos.
- Alerta para mudança abrupta na distribuição do target entre períodos.
- Alerta para queda relevante da interseção de `RA` entre anos.


