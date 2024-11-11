# UniStain
- This is built upon [huggingface/diffusers](https://github.com/huggingface/diffusers). You can find more details about Installation and Requirements.
- This is a preliminary version, more details will be coming soon.

## Training
- Prepare the dataset: (We provided the first AF-H&E dataset for virtual staining, you can [download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lshiao_connect_ust_hk/Eg1nrI3BNeNAk3pPHe0yY1IBHSpuhgkj4X3MIGmYjp837g) for research use only)
- Train a model:
```
  bash run_lora.sh
```
- Test the model:
```
  python test_wi_ref.py
```
- We also provided our trained model [here](https://drive.google.com/file/d/1v0vtl6nRH0MCKKYL1MjRzXTH4ZCBxf-x/view?usp=drive_link).
