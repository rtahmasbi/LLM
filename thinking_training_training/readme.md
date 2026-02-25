# make env
```sh
conda create -n env_thinking python=3.11
conda activate env_thinking

pip install -r requirements.txt

#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install transformers accelerate datasets peft bitsandbytes sentencepiece
#conda install -c conda-forge huggingface_hub

# add you HF token:
hf auth login

cat /workspace/.hf_home/stored_tokens

```


# use vastai for the gpu
```sh
vastai show instances
vastai search offers 'num_gpus=1 gpu_name=A100_SXM4 gpu_ram>=40 disk_space>=100 datacenter=True inet_up>500 inet_down>500 duration>3' --on-demand

vastai create instance 30178902 --template_hash 222392e8bff8afae7b41093613548200 --disk 100 --label ras1

vastai show instances
vastai start instance 31950218
vastai ssh-url 31950218
> ssh://root@80.188.223.202:11024

ssh $(vastai ssh-url 31950218)

touch ~/.no_auto_tmux



vastai destroy instance 31950218

```

