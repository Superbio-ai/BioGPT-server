import torch
from fairseq.models.transformer_lm import TransformerLanguageModel

m = TransformerLanguageModel.from_pretrained(
        "checkpoints/Pre-trained-BioGPT-Large", 
        "checkpoint.pt", 
        "data/BioGPT-Large", # change this for smaller model
        tokenizer='moses', 
        bpe='fastbpe', 
        bpe_codes="data/BioGPT-Large/bpecodes", # change this for smaller model
        min_len=100)

#Note!: If GPU memory usage error is occured, open these two lines below.It will decrease the GPU memory usage.
#print('Converting to float 16')
#m.half() # use for gpu

m.cuda()
src_tokens = m.encode("COVID-19 is")

generate = m.generate([src_tokens], beam=5)[0]
output = m.decode(generate[0]["tokens"])

print(output)
