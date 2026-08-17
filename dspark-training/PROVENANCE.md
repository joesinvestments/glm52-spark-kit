# DSpark speculator training pipeline

Source: bird/GLM-spark, recipes/4x-dspark-1m/speculator-training/ (Apache-2.0), fetched 2026-08-17.
Purpose here: retrain the DSpark draft head against OUR fleet's target hidden states (QuantTrio
Int4-Int8Mix served by our stack) so acceptance matches our quant, per bird's measured ~1.45 -> ~2.1
accepted/step for the RedHat base vs the quant-matched finetune.

Data capture in bird's flow relies on a hook in his vLLM branch (VLLM_DSPARK_CAPTURE_DIR /
VLLM_DSPARK_CAPTURE_EVERY) that records target aux hidden states + token ids per step. Our tree
does not have it; the port of that hook onto our MRv2 DSpark speculator is the first task of the
training track. Until then, b1rd/GLM-5.2-speculator.dspark-quanttrio-int4-ft (already finetuned on
this exact quant, 5 layers, block 8, markov rank 256) is the drafter under test.
