# **IQFormer: A Novel Transformer-Based Model With Multi-Modality Fusion for Automatic Modulation Recognition**

Official Code for "**IQFormer: A Novel Transformer-Based Model With Multi-Modality Fusion for Automatic Modulation Recognition**". [[paper](https://ieeexplore.ieee.org/abstract/document/10729886)]

## Citation

If our work is helpful to your research, please star us on github and cite :
> @ARTICLE{10729886,
  author={Shao, Mingyuan and Li, Dingzhao and Hong, Shaohua and Qi, Jie and Sun, Haixin},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  title={IQFormer: A Novel Transformer-Based Model With Multi-Modality Fusion for Automatic Modulation Recognition},
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Modulation;Transformers;Convolution;Time-frequency analysis;Wireless communication;Time-domain analysis;Signal to noise ratio;Market research;Interference;Automatic modulation recognition;deep learning;multi-modality fusion;transformer},
  doi={10.1109/TCCN.2024.3485118}}
> 

## **Preparation**

We conducted experiments on three datasets, namely RML2016.10a, RML2016.10b and HisarMod2019.1.

The datasets can be downloaded from the [DeepSig](https://www.deepsig.ai/datasets/)(RML2016 series), [HisarMod2019.1](https://pan.quark.cn/s/016a2f6861a2).  Special thanks to [Richardzhangxx](https://github.com/Richardzhangxx/AMR-Benchmark) for providing the .MAT file for HisarMod2019.1. For your convenience, I have combined the I/Q signals and saved them in the h5py file. If you want to know the most widely used dataset division ratio, read my paper.

```python
with h5py.File(os.path.join(args.database_path, 'HisarMod2019train.h5')) as h5file:
    train = h5file['samples'][:]
    train_label = h5file['labels'][:]
    SNR_tr = h5file['snr'][:]
		h5file.close()
with h5py.File(os.path.join(args.database_path, 'HisarMod2019test.h5')) as h5file:
    test = h5file['samples'][:]
    test_label = h5file['labels'][:]
    SNR_te = h5file['snr'][:]
    h5file.close()
```

Please extract the downloaded compressed file directly into the `./dataset` directory, or change the 

`args.database_path`.  `args.database_choose`  should keep in [2016.10a, 2016.10b, 2019].

Then you just need to 

```python
python main.py
```

If you want our pre-trained models on both three datasets, please contact **shaomy666@stu.xmu.edu.cn**

## **Environment**

These models are implemented in Keras, and the environment setting is:

- Python 3.11
- Pytorch 1.12.0
- pandas
- seaborn
- h5py
- scikit-learn
- matplotlib
- tensorboardX
- tqdm
- timm

## **License**

This code is distributed under an [MIT LICENSE](https://github.com/zjwXDU/AMC-Net/blob/main/LICENSE). Note that our code depends on other libraries and datasets which each have their own respective licenses that must also be followed.
