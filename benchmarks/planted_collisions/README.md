# Planted Subspace Fault & Collision Corpus (`benchmarks/planted_collisions/`)

Scheduled for **Stage 6 Milestone M1** (`KPI-RES-01`: `100%` recall, `<= 5%` false-alarm rate):

Contains deterministic binary `.safetensors` + PEFT `adapter_config.json` synthetic fixtures covering:
1. Dead / `NaN` poisoned weights (`RULE-SPEC-001`)
2. Rank-collapsed $r=32 \to r_{\text{eff}} \approx 1.05$ adapter (`RULE-SPEC-002`)
3. Spectral norm spike $\ge 10\times$ layer median (`RULE-SPEC-003`)
4. Colliding sign-conflicting adapter pair (`RULE-SPEC-004`)
5. Base-model mismatch & attention-only placement (`RULE-CONF-001`, `RULE-CONF-003`)
6. Hardened orthogonal control adapters (`Grade A`, `MergeRiskScore == 0.0`)
