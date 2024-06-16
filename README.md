# Quantgpt
A smaller version of GPT-2(124M) training code 

You can run the code on colab just first use the data_prep script to create the shards we are only training on 100M tokens for now

*.bf16 features requires at least a compute capability of sm_80, which is available on NVIDIA GPUs starting from the Ampere architecture so we are using fp16 in colab

## Data Dir 
use this for loading FineWeb-MINI Dataset
https://huggingface.co/datasets/AryanNsc/FineWeb-Mini

## TODO

- [ ] **Improve speed:** We are only getting 20k token/s speed in colab we have to imporve it 
- [ ] **Modify code for kaggle notebook** Modifying the code for the kaggle notebook support bc it has 2*T4 GPU
- [ ] **Saving the model** We have to save the model and add further inference caps
 .....on ...

### Contributing
Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback.

### License
MIT