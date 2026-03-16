# Related Work

## Attention kernels

The line this study measures begins with FlashAttention [@dao2022flashattention], which reorders exact
attention into a tiled, online-softmax form so that the full N x N score matrix is never materialised in
HBM. The claim needs stating precisely: the arithmetic remains quadratic in sequence length; what the
algorithm removes is the quadratic *materialisation and memory traffic*, the binding constraint on modern
GPUs. Its results are reported on NVIDIA A100. FlashAttention-2 [@dao2023flashattention2] rebalances work across thread blocks and warps and
reports 50-73% of theoretical peak FLOPs/s -- again on A100, with Hopper-specific features explicitly
deferred. FlashAttention-3 [@shah2024flashattention3] is that deferred work: warp specialisation, TMA, and
FP8 block quantisation, evaluated on H100 only, where BF16 reaches up to 840 TFLOPs/s. Flash-Decoding
[@dao2023flashdecoding] addresses decode, where q_len = 1 leaves FlashAttention's batch and query-length
parallelism unable to fill the machine; it adds a split-KV dimension with a log-sum-exp reduction, and
micro-benchmarks on A100. FlashInfer [@ye2025flashinfer] generalises this into a serving engine with
block-sparse KV storage, customisable templates and JIT specialisation, evaluated on A100 40GB SXM and
H100 80GB SXM.

Each is a strong single-device result, honest about the device it targeted; none establishes -- or set
out to -- what happens on the other capabilities a practitioner might rent. That generation-spanning
claim is assembled by readers, not made by the papers.

## Compilers versus hand-written kernels

Triton [@tillet2019triton] introduced a tile-level language and compiler intended to make such kernels
writable without CUDA, and is now the code-generation target for PyTorch's default GPU backend.
TorchInductor, described with TorchDynamo in the PyTorch 2 paper [@ansel2024pytorch2], reports a 2.27x
inference and 1.41x training geometric-mean speedup over 180+ real models -- measured on a single A100.
Earlier compiler work made portability an explicit goal: TVM [@chen2018tvm] claims performance
portability across back-ends, and Ansor [@zheng2020ansor] replaces template-guided search with
hierarchical sampling and a learned cost model. Recent evaluation efforts inherit the single-device habit.
KernelBench [@ouyang2025kernelbench] evaluates 250 PyTorch workloads, profiles on an NVIDIA L40S, and
finds LM-generated kernels beat PyTorch eager on fewer than 20% of tasks; TritonBench [@li2025tritonbench]
curates 184 real Triton operators and states its evaluations were conducted exclusively on the A100.
Whether a kernel that wins on L40S or A100 still wins on sm_90 or sm_120 is not a question these
harnesses are structured to ask.

## Serving systems built on these kernels

vLLM [@kwon2023vllm] made KV-cache paging standard, evaluated on A100 instances. Sarathi-Serve
[@agrawal2024sarathi] (chunked prefills and stall-free scheduling, on A100) and DistServe
[@zhong2024distserve] (prefill/decode disaggregation across GPUs, on nodes of eight A100-80GB) take the
prefill/decode split as their organising principle -- the two regimes measured here -- and treat the
attention kernel as a fixed, well-chosen component. Backend choice in such systems is a static preference
order over FlashAttention, FlashInfer, cuDNN and an internal Triton path, guarded by capability checks
rather than by measurements on the target device. Splitwise [@patel2024splitwise] comes closest to a
cross-generation view, characterising both phases on DGX-A100 and DGX-H100 and provisioning heterogeneous
machine pools; but that heterogeneity lives at the level of machine assignment, and the kernel beneath
each phase is still chosen by a rule nobody has shown to transfer.

## Performance portability as a field

Outside ML systems this is a mature question. The roofline model [@williams2009roofline] supplies the
per-architecture ceiling against which a kernel's transfer should be read. Pennycook et al.
[@pennycook2016metric] define performance portability and propose a metric -- a harmonic mean of
per-platform efficiency over an explicitly stated platform set -- which presumes exactly the
multi-platform matrix the attention literature does not publish. Deakin et al. [@deakin2019portability]
apply it at scale: five mini-apps, twelve platforms, six programming models, 360 combinations. Autotuning
work [@chen2018tvm; @zheng2020ansor] answers the same question by searching per target, conceding the
premise: the right implementation is a property of the target, not of the algorithm.

## What remains unaddressed

Three gaps motivate this study. First, published attention benchmarks report single-device results; the
winner is measured on one GPU and read as a property of the kernel. Across five compute-capability targets
it is not stable: the prefill winner changes on 81.9% of 144 matched configurations between A10 (sm86) and
H100 (sm90), and the decode winner on 70.9% of 299 shared cells across A10, A100 and H100. Second, they do
not verify which kernel actually dispatched: an SDPA call or a torch.compile region is reported by the
name the user typed, not by the code that ran. Here TorchInductor's fuse_attention counter was 0 on all
1215 inductor rows on every GPU -- torch.compile never rewrote naive attention into SDPA -- which no
wall-clock number alone would reveal. Third, the shipped binary set
is never treated as a variable. Under cuobjdump (no GPU required), flash-attn 2.8.3 ships SASS for sm_80, sm_90,
sm_100 and sm_120, no PTX, and nothing for sm_86 or sm_89 -- so on A10 and every Ada part it runs A100
cubins under CUDA minor-version binary compatibility -- while FlashInfer's run-time JIT cannot target
sm_120 under nvcc 12.8, failing on 950 of 960 Blackwell cells behind a misleading capability error.
The unit of performance is therefore not (kernel, GPU) but (kernel, GPU, regime, software-distribution
snapshot). The flip rates above are consistent with this packaging asymmetry as much as with
microarchitecture; separating the two would need a from-source rebuild for the missing targets, which we
did not perform, so we make no causal claim. Absolute latencies, the L40/L40S same-architecture control
that fixes a noise floor, and the NUMA-pinning correction follow.

---

## Bibliography

```bibtex
@inproceedings{dao2022flashattention,
  title     = {{FlashAttention}: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author    = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  year      = {2022},
  eprint    = {2205.14135},
  archivePrefix = {arXiv}
}

@inproceedings{dao2023flashattention2,
  title     = {{FlashAttention-2}: Faster Attention with Better Parallelism and Work Partitioning},
  author    = {Dao, Tri},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2307.08691},
  archivePrefix = {arXiv},
  note      = {arXiv preprint, July 2023}
}

@inproceedings{shah2024flashattention3,
  title     = {{FlashAttention-3}: Fast and Accurate Attention with Asynchrony and Low-precision},
  author    = {Shah, Jay and Bikshandi, Ganesh and Zhang, Ying and Thakkar, Vijay and Ramani, Pradeep and Dao, Tri},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  eprint    = {2407.08608},
  archivePrefix = {arXiv}
}

@misc{dao2023flashdecoding,
  title        = {Flash-Decoding for Long-Context Inference},
  author       = {Dao, Tri and Haziza, Daniel and Massa, Francisco and Sizov, Grigory},
  year         = {2023},
  month        = {October},
  howpublished = {PyTorch Blog},
  url          = {https://pytorch.org/blog/flash-decoding/}
}

@inproceedings{ye2025flashinfer,
  title     = {{FlashInfer}: Efficient and Customizable Attention Engine for {LLM} Inference Serving},
  author    = {Ye, Zihao and Chen, Lequn and Lai, Ruihang and Lin, Wuwei and Zhang, Yineng and Wang, Stephanie and Chen, Tianqi and Kasikci, Baris and Grover, Vinod and Krishnamurthy, Arvind and Ceze, Luis},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  year      = {2025},
  eprint    = {2501.01005},
  archivePrefix = {arXiv},
  note      = {Outstanding Paper Award}
}

@inproceedings{tillet2019triton,
  title     = {Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations},
  author    = {Tillet, Philippe and Kung, H. T. and Cox, David},
  booktitle = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL)},
  pages     = {10--19},
  year      = {2019},
  doi       = {10.1145/3315508.3329973}
}

@inproceedings{ansel2024pytorch2,
  title     = {{PyTorch} 2: Faster Machine Learning Through Dynamic {Python} Bytecode Transformation and Graph Compilation},
  author    = {Ansel, Jason and Yang, Edward and He, Horace and Gimelshein, Natalia and Jain, Animesh and Voznesensky, Michael and Bao, Bin and Bell, Peter and Berard, David and Burovski, Evgeni and Chauhan, Geeta and Chourdia, Anjali and Constable, Will and Desmaison, Alban and DeVito, Zachary and Ellison, Elias and Feng, Will and Gong, Jiong and Gschwind, Michael and Hirsh, Brian and Huang, Sherlock and Kalambarkar, Kshiteej and Kirsch, Laurent and Lazos, Michael and Lezcano, Mario and Liang, Yanbo and Liang, Jason and Lu, Yinghai and Luk, C. K. and Maher, Bert and Pan, Yunjie and Puhrsch, Christian and Reso, Matthias and Saroufim, Mark and Siraichi, Marcos Yukio and Suk, Helen and Suo, Michael and Tillet, Phil and Wang, Eikan and Wang, Xiaodong and Wen, William and Zhang, Shunting and Zhao, Xu and Zhou, Keren and Zou, Richard and Mathews, Ajit and Chanan, Gregory and Wu, Peng and Chintala, Soumith},
  booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
  year      = {2024},
  doi       = {10.1145/3620665.3640366}
}

@inproceedings{chen2018tvm,
  title     = {{TVM}: An Automated End-to-End Optimizing Compiler for Deep Learning},
  author    = {Chen, Tianqi and Moreau, Thierry and Jiang, Ziheng and Zheng, Lianmin and Yan, Eddie and Shen, Haichen and Cowan, Meghan and Wang, Leyuan and Hu, Yuwei and Ceze, Luis and Guestrin, Carlos and Krishnamurthy, Arvind},
  booktitle = {13th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  pages     = {578--594},
  year      = {2018}
}

@inproceedings{zheng2020ansor,
  title     = {Ansor: Generating High-Performance Tensor Programs for Deep Learning},
  author    = {Zheng, Lianmin and Jia, Chengfan and Sun, Minmin and Wu, Zhao and Yu, Cody Hao and Haj-Ali, Ameer and Wang, Yida and Yang, Jun and Zhuo, Danyang and Sen, Koushik and Gonzalez, Joseph E. and Stoica, Ion},
  booktitle = {14th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  pages     = {863--879},
  year      = {2020}
}

@inproceedings{ouyang2025kernelbench,
  title     = {{KernelBench}: Can {LLMs} Write Efficient {GPU} Kernels?},
  author    = {Ouyang, Anne and Guo, Simon and Arora, Simran and Zhang, Alex L. and Hu, William and R{\'e}, Christopher and Mirhoseini, Azalia},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {267},
  pages     = {47356--47415},
  year      = {2025},
  eprint    = {2502.10517},
  archivePrefix = {arXiv}
}

@inproceedings{li2025tritonbench,
  title     = {{TritonBench}: Benchmarking Large Language Model Capabilities for Generating {Triton} Operators},
  author    = {Li, Jianling and Li, Shangzhan and Gao, Zhenye and others},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {23053--23066},
  address   = {Vienna, Austria},
  year      = {2025},
  eprint    = {2502.14752},
  archivePrefix = {arXiv}
}

@inproceedings{kwon2023vllm,
  title     = {Efficient Memory Management for Large Language Model Serving with {PagedAttention}},
  author    = {Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)},
  year      = {2023},
  doi       = {10.1145/3600006.3613165}
}

@inproceedings{agrawal2024sarathi,
  title     = {Taming Throughput-Latency Tradeoff in {LLM} Inference with {Sarathi-Serve}},
  author    = {Agrawal, Amey and Kedia, Nitin and Panwar, Ashish and Mohan, Jayashree and Kwatra, Nipun and Gulavani, Bhargav S. and Tumanov, Alexey and Ramjee, Ramachandran},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year      = {2024},
  eprint    = {2403.02310},
  archivePrefix = {arXiv}
}

@inproceedings{zhong2024distserve,
  title     = {{DistServe}: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving},
  author    = {Zhong, Yinmin and Liu, Shengyu and Chen, Junda and Hu, Jianbo and Zhu, Yibo and Liu, Xuanzhe and Jin, Xin and Zhang, Hao},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year      = {2024},
  eprint    = {2401.09670},
  archivePrefix = {arXiv}
}

@inproceedings{patel2024splitwise,
  title     = {{Splitwise}: Efficient Generative {LLM} Inference Using Phase Splitting},
  author    = {Patel, Pratyush and Choukse, Esha and Zhang, Chaojie and Shah, Aashaka and Goiri, {\'I}{\~n}igo and Maleki, Saeed and Bianchini, Ricardo},
  booktitle = {Proceedings of the 51st Annual International Symposium on Computer Architecture (ISCA)},
  year      = {2024},
  eprint    = {2311.18677},
  archivePrefix = {arXiv}
}

@article{williams2009roofline,
  title   = {Roofline: An Insightful Visual Performance Model for Multicore Architectures},
  author  = {Williams, Samuel and Waterman, Andrew and Patterson, David},
  journal = {Communications of the ACM},
  volume  = {52},
  number  = {4},
  pages   = {65--76},
  year    = {2009},
  doi     = {10.1145/1498765.1498785}
}

@misc{pennycook2016metric,
  title         = {A Metric for Performance Portability},
  author        = {Pennycook, S. J. and Sewall, J. D. and Lee, V. W.},
  year          = {2016},
  eprint        = {1611.07409},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PF}
}

@inproceedings{deakin2019portability,
  title     = {Performance Portability across Diverse Computer Architectures},
  author    = {Deakin, Tom and McIntosh-Smith, Simon and Price, James and Poenaru, Andrei and Atkinson, Patrick and Popa, Codrin and Salmon, Justin},
  booktitle = {2019 IEEE/ACM International Workshop on Performance, Portability and Productivity in HPC (P3HPC)},
  year      = {2019},
  doi       = {10.1109/P3HPC49587.2019.00006}
}
```
