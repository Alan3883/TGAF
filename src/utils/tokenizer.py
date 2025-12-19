import sentencepiece as spm
from typing import List


class BPETokenizer:
    """
    SentencePiece-based BPE tokenizer for caption encoding/decoding.
    """

    def __init__(self, model_path: str, bos="<bos>", eos="<eos>", pad="<pad>"):
        """
        Args:
            model_path: Path to SentencePiece model file
            bos: Beginning-of-sequence token
            eos: End-of-sequence token
            pad: Padding token
        """
        
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.bos, self.eos, self.pad = bos, eos, pad
        self.bos_id = self.sp.piece_to_id(bos)
        self.eos_id = self.sp.piece_to_id(eos)
        self.pad_id = self.sp.piece_to_id(pad)
    
    def encode(self, text: str):
        """
        Encode text to token IDs with BOS and EOS tokens.
        Args:
            text: Input text string
        Returns:
            Object with .ids attribute containing list of token IDs
        """

        ids = [self.bos_id] + self.sp.encode(text, out_type=int) + [self.eos_id]
        
        class TokenIds:
            def __init__(self, ids):
                self.ids = ids
        return TokenIds(ids)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text string.
        Args:
            token_ids: List of token IDs
        Returns:
            Decoded text string
        """

        # Filter out special tokens
        filtered_ids = [tid for tid in token_ids if tid not in [self.bos_id, self.eos_id, self.pad_id]]
        return self.sp.decode(filtered_ids)
    
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.sp.vocab_size()