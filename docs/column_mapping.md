# Tabela de Mapeamento de Colunas Equivalentes

Esta tabela define o crosswalk de colunas equivalentes entre anos (`2022`, `2023`, `2024`).
O objetivo é garantir uma fonte única de verdade para harmonização de schema antes do alinhamento.

## Regras

- A harmonização usa prioridade de aliases por ano.
- Quando múltiplos aliases para a mesma coluna canônica aparecem no mesmo DataFrame, a resolução usa `combine_first` por prioridade.
- Apenas a coluna canônica permanece no schema final para os campos mapeados.
- O processo gera auditoria agregada (`column_mapping_report`) sem valores de células ou identificadores.

## Crosswalk

| Canônica | Aliases 2022 (prioridade) | Aliases 2023 (prioridade) | Aliases 2024 (prioridade) | Notas |
|---|---|---|---|---|
| `Defasagem` | `Defasagem`, `Defasagem__dup1`, `Defasagem.1`, `Defas` | `Defasagem`, `Defasagem__dup1`, `Defasagem.1`, `Defas` | `Defasagem`, `Defasagem__dup1`, `Defasagem.1`, `Defas` | Resolve duplicatas com `combine_first` |
| `Mat` | `Matem`, `Mat` | `Mat` | `Mat` | Nota de matemática |
| `Por` | `Portug`, `Por` | `Por` | `Por` | Nota de português |
| `Ing` | `Inglês`, `Ing` | `Ing` | `Ing` | Nota de inglês |
| `Data_Nasc` | `Ano nasc`, `Data_Nasc` | `Data de Nasc`, `Data_Nasc` | `Data de Nasc`, `Data_Nasc` | `Ano nasc` (2022) é harmonizado para o header canônico |
| `Fase_Ideal` | `Fase ideal`, `Fase_Ideal` | `Fase Ideal`, `Fase_Ideal` | `Fase Ideal`, `Fase_Ideal` | Variações de capitalização |
| `Nome_Anon` | `Nome`, `Nome_Anon` | `Nome Anonimizado`, `Nome_Anon` | `Nome Anonimizado`, `Nome_Anon` | Harmonização de header; não implica anonimização semântica |
| `Idade` | `Idade 22`, `Idade` | `Idade` | `Idade` | Padronização do campo de idade |

Implementação de referência: `src/column_mapping.py`.
