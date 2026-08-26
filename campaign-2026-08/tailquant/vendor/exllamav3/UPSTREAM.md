# Vendored: exllamav3 trellis quantizer (subset)

Origin: https://github.com/turboderp-org/exllamav3 (MIT, LICENSE included verbatim).
Vendored 2026-08-25 by the GLM-5.2 collab campaign.

Contents:
- modules-quant/  <- exllamav3/exllamav3/modules/quant/ (python driver: LDLQ pipeline,
  tile quantization dispatch, MCG/MUL1/3INST codebook constants incl.
  codebook_mcg_mult = 0xCBAC1FED - same multiplier as b12x `mcg`)
- ext-quant-src/  <- exllamav3/exllamav3/exllamav3_ext/quant/ (CUDA reference for
  decode_3inst lop3 decode law; CPU adaptation targets this math)

Credit rule: turboderp / turboderp-org credit stays ahead of any collab
modification. Modifications live in port/, never in this tree.
