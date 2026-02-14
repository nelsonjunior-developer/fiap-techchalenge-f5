# Data Contract 2024

| Coluna | DType | Presence | PII | Regras |
|---|---|---|---|---|
| Ano ingresso | Int64 | original | no | dtype:error, missing:error, domain:error |
| Atingiu PV | string | original | no | dtype:error, missing:info, domain:warning |
| Ativo/ Inativo | string | original | no | dtype:error, missing:warning, domain:warning |
| Ativo/ Inativo__dup1 | string | original | no | dtype:error, missing:warning, domain:info |
| Avaliador1 | string | original | yes | dtype:error, missing:info, domain:info |
| Avaliador2 | string | original | yes | dtype:error, missing:info, domain:info |
| Avaliador3 | string | original | yes | dtype:error, missing:info, domain:info |
| Avaliador4 | string | original | yes | dtype:error, missing:info, domain:info |
| Avaliador5 | string | original | yes | dtype:error, missing:info, domain:info |
| Avaliador6 | string | original | yes | dtype:error, missing:info, domain:info |
| Cf | Float64 | original | no | dtype:error, missing:info, domain:info |
| Cg | Float64 | original | no | dtype:error, missing:info, domain:info |
| Ct | Float64 | original | no | dtype:error, missing:info, domain:info |
| Data_Nasc | datetime64[ns] | original | no | dtype:error, missing:warning, domain:warning |
| Defasagem | Int64 | original | no | dtype:error, missing:error, domain:error |
| Destaque IDA | string | original | no | dtype:error, missing:info, domain:info |
| Destaque IEG | string | original | no | dtype:error, missing:info, domain:info |
| Destaque IPV | string | original | no | dtype:error, missing:info, domain:info |
| Destaque IPV__dup1 | string | structural_optional | no | dtype:error, missing:info, domain:info |
| Escola | string | original | no | dtype:error, missing:warning, domain:info |
| Fase | string | original | no | dtype:error, missing:warning, domain:info |
| Fase_Ideal | string | original | no | dtype:error, missing:warning, domain:info |
| Gênero | string | original | no | dtype:error, missing:error, domain:error |
| IAA | Float64 | original | no | dtype:error, missing:warning, domain:error |
| IAN | Float64 | original | no | dtype:error, missing:warning, domain:error |
| IDA | Float64 | original | no | dtype:error, missing:warning, domain:error |
| IEG | Float64 | original | no | dtype:error, missing:warning, domain:error |
| INDE | Float64 | original | no | dtype:error, missing:warning, domain:error |
| INDE 2023 | Float64 | structural_optional | no | dtype:error, missing:info, domain:error |
| INDE 2024 | Float64 | original | no | dtype:error, missing:info, domain:error |
| INDE 22 | Float64 | original | no | dtype:error, missing:info, domain:error |
| INDE 23 | Float64 | original | no | dtype:error, missing:info, domain:error |
| IPP | Float64 | original | no | dtype:error, missing:warning, domain:error |
| IPS | Float64 | original | no | dtype:error, missing:warning, domain:error |
| IPV | Float64 | original | no | dtype:error, missing:warning, domain:error |
| Idade | Int64 | original | no | dtype:error, missing:error, domain:error |
| Indicado | string | original | no | dtype:error, missing:info, domain:warning |
| Ing | Float64 | original | no | dtype:error, missing:info, domain:error |
| Instituição de ensino | string | original | no | dtype:error, missing:warning, domain:info |
| Mat | Float64 | original | no | dtype:error, missing:warning, domain:error |
| Nome_Anon | string | original | yes | dtype:error, missing:info, domain:info |
| Nº Av | Int64 | original | no | dtype:error, missing:error, domain:error |
| Pedra 20 | string | original | no | dtype:error, missing:info, domain:warning |
| Pedra 2023 | string | structural_optional | no | dtype:error, missing:info, domain:warning |
| Pedra 2024 | string | original | no | dtype:error, missing:info, domain:warning |
| Pedra 21 | string | original | no | dtype:error, missing:info, domain:warning |
| Pedra 22 | string | original | no | dtype:error, missing:info, domain:warning |
| Pedra 23 | string | original | no | dtype:error, missing:info, domain:warning |
| Pedra_Ano | string | original | no | dtype:error, missing:info, domain:warning |
| Por | Float64 | original | no | dtype:error, missing:warning, domain:error |
| RA | string | original | yes | dtype:error, missing:error, domain:info |
| Rec Av1 | string | original | no | dtype:error, missing:info, domain:info |
| Rec Av2 | string | original | no | dtype:error, missing:info, domain:info |
| Rec Av3 | string | structural_optional | no | dtype:error, missing:info, domain:info |
| Rec Av4 | string | structural_optional | no | dtype:error, missing:info, domain:info |
| Rec Psicologia | Float64 | original | no | dtype:error, missing:info, domain:info |
| Turma | string | original | no | dtype:error, missing:warning, domain:info |
