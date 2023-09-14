CONVERSER: Few-Shot Conversational Dense Retrieval with Synthetic Data Generation
===
[SIGDIAL 2023 paper](https://arxiv.org/abs/2309.06748)

## Reference
Please cite the following paper
```
    @inproceedings{huang-etal-2022-plm,
        title = "CONVERSER: Few-Shot Conversational Dense Retrieval With Synthetic Data Generation",
        author = "Huang, Chao-Wei and Hsu, Chen-Yu and Hsu, Tsu-Yuan and Li, Chen-An and Chen, Yun-Nung",
        booktitle = "Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
        month = sep,
        year = "2023",
        address = "Prague, Czech Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://doi.org/10.48550/arXiv.2309.06748"
    }
```

## Requirements
* Python >= 3.6
* Install the required Python packages with `pip3 install -r requirements.txt`

## Datasets
Our generated dataset can be found in the [google drive](https://drive.google.com/drive/folders/1z375Z5-3vNnB6Pi37I0u-P1BDDW_9kDB?usp=sharing)

## How to run
### Pretrained LLM
We used LLaMA-13B in our experiments. Please apply for access [here](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform). You can also try other open-source LLMs such as [LLaMA-2](https://ai.meta.com/llama/) and [Falcon](https://huggingface.co/blog/falcon). Note that our method doesn't require instruction-tuned LLMs, so you can use any pretrained LLM.

### Corpus
In order to run dialogue generation, you'll need a collection of passages. In our experiments, we used the passage collection from [OR-QuAC](https://github.com/prdwb/orconvqa-release). You can process the released data with the [ConvDR](https://github.com/thunlp/ConvDR) repo.

### Run the generation script
- Modify the paths to *LLAMA_CHECKPOINT_DIR* and *COLLECTION_JSONL* in `generate_dialog.py` to your local paths.
- Simple run
```
    python3 generate_dialog.py
```
- You can also find our generated datasets [here](https://drive.google.com/drive/folders/1z375Z5-3vNnB6Pi37I0u-P1BDDW_9kDB?usp=sharing)

### Training a DPR model
Please refer to the [original DPR repo](https://github.com/facebookresearch/DPR) or the more resource-light implementation [GC-DPR](https://github.com/luyug/GC-DPR) for training a DPR model given the generated dataset. With **GC-DPR**, you should be able to train a DPR model with only 1 GPU. Below is a reference command we used with **GC-DPR** to train the model:
```
CUDA_VISIBLE_DEVICES=0 python3 train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-uncased \
    --seed 12345 \
    --sequence_length 384 \
    --warmup_steps 1237 \
    --batch_size 64 \
    --dev_batch_size 16 \
    --do_lower_case \
    --train_file ${GENERATED_DATASET} \
    --dev_file ../ConvDR/datasets/or-quac/dev_dpr.json \
    --output_dir ${MODEL_DIR} \
    --learning_rate 2e-05 \
    --num_train_epochs 30 \
    --val_av_rank_start_epoch 0 \
    --fp16 \
    --grad_cache \
    --q_chunk_size 8 \
    --ctx_chunk_size 8 \
    --global_loss_buf_sz 2097152 \
    --val_av_rank_max_qs 10000
``` 