"""
Build vocabulary from MSR-VTT captions using SentencePiece BPE tokenization.
"""
import argparse
import json
import os
import glob
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser(description="Build BPE vocabulary from MSR-VTT captions")
    ap.add_argument("--splits", default="data/msrvtt/splits", help="Directory containing split JSON files")
    ap.add_argument("--out", default="outputs/spm/bpe16k.model", help="Output SentencePiece model path")
    ap.add_argument("--vocab_size", type=int, default=16000, help="Vocabulary size")
    args = ap.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tmp_txt = "outputs/spm/corpus.txt"
    os.makedirs(os.path.dirname(tmp_txt), exist_ok=True)
    
    # Collect all captions from split files
    print(f"Collecting captions from {args.splits}...")
    with open(tmp_txt, "w", encoding="utf-8") as f:
        json_files = glob.glob(os.path.join(args.splits, "*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {args.splits}")
        
        for fp in json_files:
            print(f"Processing {fp}...")
            with open(fp, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                # Handle both list and dict formats
                if isinstance(data, list):
                    items = data
                else:
                    items = list(data.values()) if isinstance(data, dict) else []
                
                for rec in items:
                    caption = rec.get("caption", "")
                    # Handle caption as string or list
                    if isinstance(caption, list):
                        for cap in caption:
                            cap = cap.strip().lower()
                            if cap:
                                f.write(cap + "\n")
                    else:
                        caption = caption.strip().lower()
                        if caption:
                            f.write(caption + "\n")
        
        # Add special tokens
        f.write("<bos>\n<eos>\n<pad>\n")
    
    print(f"Training SentencePiece model with vocab_size={args.vocab_size}...")
    # Train SentencePiece model
    spm.SentencePieceTrainer.Train(
        input=tmp_txt,
        model_prefix=args.out[:-6],  # Remove .model extension
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        user_defined_symbols=["<bos>", "<eos>", "<pad>"]
    )
    
    print(f"Saved SentencePiece model to {args.out}")
    print(f"Cleaning up temporary file: {tmp_txt}")
    os.remove(tmp_txt)


if __name__ == "__main__":
    main()