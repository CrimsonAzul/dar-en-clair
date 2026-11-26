"""Fine-tuning LoRA de bge-m3 (recherche dense) sur les triplets de donnees.py.

Choix pour une carte de 6 Go (RTX 3060 portable) :
- LoRA r=16 sur les couches d'attention et denses : environ 1 % des poids sont entraines, le
  modele de base reste fige en bf16 (1,1 Go) ;
- CachedMultipleNegativesRankingLoss (GradCache) : lot effectif de 64 questions (64 positifs +
  64 negatifs difficiles + les autres passages du lot comme negatifs), calcule par mini-lots de 16 ;
- gradient checkpointing : les activations sont recalculees pendant la retropropagation au lieu
  d'etre gardees. Sans lui, le pic depasse 6 Go et Windows deborde en silence sur la RAM :
  130 s par pas au lieu d'une vingtaine ;
- plafond de VRAM a 93 % : mieux vaut une erreur claire qu'un debordement qui ralentit tout ;
- scale = 50 (temperature 0,02), celle de l'entrainement d'origine de bge-m3 ;
- textes coupes a 512 tokens.

Usage : python -m darenclair.finetune.entrainer [--epoques 2] [--limite 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
MODELE_BASE = "BAAI/bge-m3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoques", type=float, default=2)
    ap.add_argument("--lot", type=int, default=64)
    ap.add_argument("--mini-lot", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rang", type=int, default=16)
    ap.add_argument("--limite", type=int, default=0, help="nombre max de triplets (essai rapide)")
    ap.add_argument("--sortie", type=Path, default=RACINE / "models" / "bge-m3-lora")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    torch.cuda.set_per_process_memory_fraction(0.93)
    lignes = [json.loads(l) for l in (RACINE / "data" / "finetune" / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if a.limite:
        lignes = lignes[: a.limite]
    ds = Dataset.from_list(lignes)

    model = SentenceTransformer(MODELE_BASE, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 512
    model.add_adapter(LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=a.rang, lora_alpha=2 * a.rang, lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"],
    ))
    # use_reentrant=False : sinon aucun gradient ne remonte, l'entree des couches figees n'en demandant pas.
    model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Poids LoRA en float32 (le modele fige reste en bf16) : en bf16, une mise a jour d'Adam
    # (~1e-4) sur un poids de ~0,03 est sous la precision (2^-8) et serait arrondie a zero.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    entraines = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"parametres entraines : {entraines / 1e6:.1f} M sur {total / 1e6:.0f} M ({100 * entraines / total:.2f} %)")

    loss = CachedMultipleNegativesRankingLoss(model, scale=50.0, mini_batch_size=a.mini_lot)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(RACINE / "models" / "run"),
        num_train_epochs=a.epoques,
        per_device_train_batch_size=a.lot,
        learning_rate=a.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # pas deux fois le meme passage dans un lot
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=0,
        dataloader_num_workers=0,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    torch.cuda.reset_peak_memory_stats()
    debut = time.time()
    trainer.train()
    print(f"duree : {(time.time() - debut) / 60:.1f} min | pic VRAM : {torch.cuda.max_memory_allocated() / 2**30:.2f} Go")

    a.sortie.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(str(a.sortie))  # n'ecrit que l'adaptateur LoRA (quelques Mo)
    print(f"adaptateur enregistre dans {a.sortie}")


if __name__ == "__main__":
    main()
    # Sous Windows, le processus reste bloque a la fermeture apres usage de CUDA : on sort net.
    sys.stdout.flush()
    os._exit(0)
