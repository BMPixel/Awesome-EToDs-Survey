# Awesome-EToDs-Survey

🔥 **Collection of papers, benchmarks and newest trends in the domain of End-to-end ToDs**

🌟 **Any contributions via PRs, issues, emails or other methods are greatly appreciated.**

🔮 **Interactive paperlist&benchmark website is also available on [etods.net](https://etods.net)**

## Intro: What is the End-to-end ToDs (EToDs)?

In contrast to traditional pipelined task-oriented dialogue systems(ToDs), EToDs are able to directly map user utterances to system actions without any intermediate representations.  EToDs are usually trained in an end-to-end manner, which means that the model is trained to directly map user utterances to system actions without any intermediate representations. 

In this repo, we further categorize EToDs into three types, as shown in the following figure.

- Modularly end-to-end task-oriented dialogue with out pretraining models
- Modularly end-to-end task-oriented dialogue with pretraining models
- Fully end-to-end task-oriented dialogue

<div align=center><img src="./assets/etods.png" width="80%" /></div>

## Table of Content (ToC)

## 1. Modularly EToD

Modularly EToDs are systems that generate responses using modularized components which are trained in an end-to-end manner, differing from fully EToDs by their non-differentiable API call knowledge base retrieval

### 1.1 Modularly EToD w/o PLM

Modularly EToD without Pretrained Language Models (PLMs) primarily focuses on optimizing dialogue systems with either supervised learning or reinforcement learning techniques

- [2020] **A Probabilistic End-To-End Task-Oriented Dialog Model with Latent Belief States towards Semi-Supervised Learning .** *Zhang et al EMNLP.* [[paper]](https://aclanthology.org/2020.emnlp-main.740/) [[code]](https://github.com/thu-spmi/LABES)
- [2020] **Attention over Parameters for Dialogue Systems .** *Madotto et al NeurIPS.* [[paper]](https://arxiv.org/abs/2001.01871)
- [2020] **LAVA: Latent Action Spaces via Variational Auto-encoding for Dialogue Policy Optimization .** *Lubis et al COLING.* [[paper]](https://aclanthology.org/2020.coling-main.41/)
- [2020] **SUMBT+LaRL: Effective Multi-Domain End-to-End Neural Task-Oriented Dialog System .** *Lee et al IEEE.* [[paper]](https://arxiv.org/abs/2009.10447)
- [2020] **UniConv: A Unified Conversational Neural Architecture for Multi-domain Task-oriented Dialogues .** *Le et al EMNLP.* [[paper]](https://virtual.2020.emnlp.org/paper_main.1012.html) [[code]](https://github.com/henryhungle/UniConv)
- [2019] **A Modular Task-oriented Dialogue System Using a Neural Mixture-of-Experts .** *Pei et al WCIS.* [[paper]](https://arxiv.org/abs/1907.05346)
- [2019] **Flexibly-Structured Model for Task-Oriented Dialogues.** *Shu et al SIGDIAL.* [[paper]](https://aclanthology.org/W19-5922/) [[code]](https://github.com/uber-research/FSDM)
- [2019] **Incremental Learning from Scratch for Task-Oriented Dialogue Systems.** *Wang et al ACL.* [[paper]](https://aclanthology.org/P19-1361.pdf) [[code]](https://github.com/Leechikara/Incremental-Dialogue-System)
- [2019] **Learning End-to-End Goal-Oriented Dialog with Maximal User Task Success and Minimal Human Agent Use.** * Rajendran et al TACL.* [[paper]](https://aclanthology.org/Q19-1024.pdf) [[code]](https://github.com/IBM/modified-bAbI-dialog-tasks)
- [2019] **MOSS: End-to-End Dialog System Framework with Modular Supervision .** *Liang et al AAAI.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6349/6205)
- [2019] **Rethinking Action Spaces for Reinforcement Learning in End-to-end Dialog Agents with Latent Variable Models .** *Zhao et al NAACL.* [[paper]](https://aclanthology.org/N19-1123.pdf) [[code]](https://github.com/snakeztc/NeuralDialog-LaRL)
- [2019] **Structured Fusion Networks for Dialog.** *Mehri et al SIGDIAL.* [[paper]](https://aclanthology.org/W19-5921)
- [2019] **Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context.** *Zhang et al AAAI.* [[paper]](https://arxiv.org/abs/1911.10484) [[code]](https://github.com/thu-spmi/damd-multiwoz)
- [2018] **Dialogue Learning with Human Teaching and Feedback in End-to-End Trainable Task-Oriented Dialogue Systems .** *Liu et al NAACL.* [[paper]](https://aclanthology.org/N18-1187/)
- [2018] **End-to-End Learning of Task-Oriented Dialogs .** *Liu and Lane NAACL.* [[paper]](https://aclanthology.org/N18-4010/)
- [2018] **Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures.** *Lei et al ACL.* [[paper]](https://aclanthology.org/P18-1133/) [[code]](https://github.com/WING-NUS/sequicity)
- [2017] **A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue .** *Eric and Manning EACL.* [[paper]](https://aclanthology.org/E17-2075/)
- [2017] **An End-to-End Trainable Neural Network Model with Belief Tracking for Task-Oriented Dialog .** *Liu and Lane InterSpeech.* [[paper]](https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/1326.PDF)
- [2017] **End-to-End Optimization of Task-Oriented Dialogue Model with Deep Reinforcement Learning.** *Liu et al Arxiv.* [[paper]](https://arxiv.org/abs/1711.10712)
- [2017] **End-to-End Task-Completion Neural Dialogue Systems .** *Li et al IJCNLP.* [[paper]](https://aclanthology.org/I17-1074.pdf) [[code]](https://github.com/MiuLab/TC-Bot)
- [2017] **Generative Encoder-Decoder Models for Task-Oriented Spoken Dialog Systems with Chatting Capability.** *Zhao et al SIGDIAL.* [[paper]](https://aclanthology.org/W17-5505/)
- [2017] **Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning .** *Williams et al ACL.* [[paper]](https://aclanthology.org/P17-1062/)
- [2016] **A network- based end-to-end trainable task-oriented dialogue system.** *Wen et al EACL.* [[paper]](https://aclanthology.org/E17-1042.pdf) [[code]](NDM)
- [2016] **Towards End-to-End Learning for Dialog State Tracking and Management using Deep Reinforcement Learning.** *Zhao and Eskenazi SIGDIAL.* [[paper]](https://aclanthology.org/W16-3601/) [[code]](https://github.com/snakeztc/NeuralDialog-DM)

### 1.2 Modularly EToD w/ PLM

Modularly EToD with PLM incorporates Pretrained Language Models using either decoder-only PLMs like GPT-2, which takes dialogue context, belief state, and database state as input to generate system responses, or encoder-decoder PLMs​

- [2023] **A Preliminary Evaluation of ChatGPT for Zero-shot Dialogue Understanding.** *Pan et al ArXiv.* [[paper]](https://api.semanticscholar.org/CorpusID:258049061)
- [2023] **ChatGPT for Zero-shot Dialogue State Tracking: A Solution or an Opportunity?.** *Heck et al ArXiv.* [[paper]](https://api.semanticscholar.org/CorpusID:259063822)
- [2023] **Are Large Language Models All You Need for Task-Oriented Dialogue?.** *Hudevcek and Dusek SIGDIAL.* [[paper]](https://api.semanticscholar.org/CorpusID:258108409)
- [2022] **Autoregressive Entity Generation for End-to-End Task-Oriented Dialog.** *Huang et al COLING.* [[paper]](https://aclanthology.org/2022.coling-1.25.pdf)
- [2022] **BORT: Back and Denoising Reconstruction for End-to-End Task-Oriented Dialog.** *Sun et al NAACL.* [[paper]](https://aclanthology.org/2022.findings-naacl.166.pdf) [[code]](https://github.com/jd-ai-research-nlp/bort)
- [2022] **SPACE-3: Unified Dialog Model Pre-training for Task-Oriented Dialog Understanding and Generation.** *He et al SIGIR.* [[paper]](https://arxiv.org/pdf/2209.06664) [[code]](https://github.com/alibabaresearch/damo-convai)
- [2022] **Task-Oriented Dialogue System as Natural Language Generation .** *Wang et al SIGIR.* [[paper]](https://arxiv.org/pdf/2108.13679) [[code]](https://github.com/victorwz/tod_as_nlg)
- [2022] **Q-TOD: A Query-driven Task-oriented Dialogue System.** *Tian et al EMNLP.* [[paper]](https://arxiv.org/abs/2210.07564) [[code]](https://github.com/PaddlePaddle/Knover/tree/develop/projects/Q-TOD)
- [2021] **[CASPI] Causal-aware Safe Policy Improvement for Task-oriented Dialogue.** *Ramachandran et al  ACL.* [[paper]](https://aclanthology.org/2022.acl-long.8.pdf) [[code]](https://github.com/salesforce/CASPI)
- [2021] **AuGPT: Auxiliary Tasks and Data Augmentation for End-To-End Dialogue with Pre-Trained Language Models .** *Kulha_nek et al EMNLP.* [[paper]](https://aclanthology.org/2021.nlp4convai-1.19.pdf) [[code]](https://github.com/ufal/augpt)
- [2021] **GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-Supervised Learning and Explicit Policy Injection .** *He et al AAAI.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/download/21320/version/19607/21069) [[code]](https://github.com/siat-nlp/GALAXY)
- [2021] **Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System .** *Su et al AAAI.* [[paper]](https://aclanthology.org/2022.acl-long.319.pdf) [[code]](https://github.com/awslabs/pptod)
- [2021] **Improving End-to-End Task-Oriented Dialog System with A Simple Auxiliary Task.** *Lee Findings.* [[paper]](https://aclanthology.org/2021.findings-emnlp.112)
- [2021] **A Co-Interactive Transformer for Joint Slot Filling and Intent Detection.** *Qin et al ICASSP.* [[paper]](https://doi.org/10.1109/ICASSP39728.2021.9414110) [[code]](https://github. com/kangbrilliant/DCA-Net)
- [2020] **A Simple Language Model for Task-Oriented Dialogue.** *Hosseini-Asl et al NeurIPS.* [[paper]](https://proceedings.neurips.cc/paper/2020/file/e946209592563be0f01c844ab2170f0c-Paper.pdf) [[code]](https://github.com/salesforce/simpletod)
- [2020] **End-to-End Neural Pipeline for Goal-Oriented Dialogue Systems using GPT-2.** *Ham et al ACL.* [[paper]](https://aclanthology.org/2020.acl-main.54.pdf)
- [2020] **End-to-End Trainable Non-Collaborative Dialog System .** *Li et al AAAI.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6345/6201) [[code]](https://gitlab.com/ucdavisnlp/antiscam)
- [2020] **MinTL: Minimalist Transfer Learning for Task-Oriented Dialogue Systems .** *Lin et al EMNLP.* [[paper]](https://aclanthology.org/2020.emnlp-main.273.pdf) [[code]](https://github.com/zlinao/MinTL)
- [2020] **Soloist : BuildingTask Bots at Scale with Transfer Learning and Machine Teaching.** *Peng et al TACL.* [[paper]](https://aclanthology.org/2021.tacl-1.49.pdf) [[code]](https://aka.ms/soloist. )
- [2020] **UBAR: Towards Fully End-to-End Task-Oriented Dialog Systems with GPT-2.** *Yang et al AAAI.* [[paper]](https://arxiv.org/pdf/2012.03539.pdf) [[code]](https://github.com/TonyNemo/UBAR-MultiWOZ)
- [2020] **AGIF: An Adaptive Graph-Interactive Framework for Joint Multiple Intent Detection and Slot Filling.** *Qin et al EMNLP.* [[paper]](https://aclanthology.org/2020.findings-emnlp.163) [[code]](https://github.com/LooperXX/AGIF)
- [2019] **Alternating Recurrent Dialog Model with Large-scale Pre-trained Language Models .** *Wu et al EACL.* [[paper]](https://aclanthology.org/2021.eacl-main.110.pdf) [[code]](https://github.com/qywu/ARDM )
- [2019] **Hello, It's GPT-2 - How Can I Help You? Towards the Use of Pretrained Language Models for Task-Oriented Dialogue Systems.** *Budzianowski et al ACL.* [[paper]](https://aclanthology.org/D19-5602.pdf)

## 2. Fully EToD

Fully EToD systems incorporate knowledge bases directly, using neural networks to query the knowledge base in a differentiable manner, which allows for end-to-end training without the need for intermediate modular annotations

<div align=center><img src="./assets/fully.png" width="80%" /></div>

### 2.1 Entity Triplet Representation

Entity Triplet Representation in Fully EToD stores knowledge base entities in a triplet format (subject, relation, object), which is summed up through word embeddings, offering a widely used method for representing knowledge base entities

- [2021] **Intention Reasoning Network for Multi-Domain End-to-end Task-Oriented Dialogue.** *Ma et al EMNLP.* [[paper]](https://aclanthology.org/2021.emnlp-main.174.pdf)
- [2021] **Intention Reasoning Network for Multi-Domain End-to-end Task-Oriented Dialogue.** *Ma et al EMNLP.* [[paper]](https://aclanthology.org/2021.emnlp-main.174)
- [2020] **Dual Dynamic Memory Network for End-to-End Multi-turn Task-oriented Dialog Systems .** *Wang et al COLING.* [[paper]](https://aclanthology.org/2020.coling-main.362.pdf) [[code]](https://github.com/siat-nlp/ddmn)
- [2020] **Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog.** *Qin et al ACL.* [[paper]](https://aclanthology.org/2020.acl-main.565.pdf) [[code]](https://github.com/LooperXX/DF-Net)
- [2019] **Disentangling Language and Knowledge in Task-Oriented Dialogs.** *Raghu et al NAACL.* [[paper]](https://aclanthology.org/N19-1126/) [[code]](https://github.com/dair-iitd/BossNet)
- [2019] **Global-to-local Memory Pointer Networks for Task-Oriented Dialogue.** *Wu et al ICLR.* [[paper]](https://arxiv.org/abs/1901.04713) [[code]](https://github.com/jasonwu0731/GLMP)
- [2019] **A Working Memory Model for Task-oriented Dialog Response Generation.** *Chen tal ACL.* [[paper]](https://aclanthology.org/P19-1258)
- [2018] **Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems .** *Madotto et al ACL.* [[paper]](https://aclanthology.org/P18-1136/) [[code]](https://github.com/HLTCHKUST/Mem2Seq)

### 2.2 Row-level Representation

Row-level Representation in Fully EToD addresses the limitations of triplet representation by considering the relationships across entities within the same row of a knowledge base, allowing for a more nuanced retrieval of relevant KB rows and columns

- [2022] **A Hierarchical Memory Model for Task-Oriented Dialogue System.** *Zeng et al IEICE.* [[paper]](https://www.jstage.jst.go.jp/article/transinf/E105.D/8/E105.D_2022EDP7001/_article) [[code]](https://github.com/zengyazy/HM2Seq)
- [2021] **Constraint based Knowledge Base Distillation in End-to-End Task Oriented Dialogs.** *Raghu et al IJCNLP.* [[paper]](https://aclanthology.org/2021.findings-acl.448) [[code]](https://github.com/dair-iitd/CDNet)
- [2019] **Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever .** *Qin et al EMNLP.* [[paper]](https://aclanthology.org/D19-1013.pdf) [[code]](https://github.com/yizhen20133868/Retriever-Dialogue)
- [2018] **Multi-Level Memory for Task Oriented Dialogs .** *Reddy et al NAACL.* [[paper]](https://aclanthology.org/N19-1375/) [[code]](https://github.com/DineshRaghu/multi-level-memory-network)
- [2018] **Sequence-to-Sequence Learning for Task-oriented Dialogue with Dialogue State Representation .** *Wen et al COLING.* [[paper]](http://ir.hit.edu.cn/~car/papers/coling18-hywen.pdf)
- [2017] **Towards End-to-End Reinforcement Learning of Dialogue Agents for Information Access .** *Dhingra et al ACL.* [[paper]](https://arxiv.org/abs/1609.00777) [[code]](https://github.com/MiuLab/KB-InfoBot)

### 2.3 Graph Representation

Graph Representation in Fully EToD aims to enhance the contextualization of entity embeddings within a knowledge base by densely linking entities to related slot titles in the dialogue history, utilizing graph-based reasoning or attention mechanisms for a more integrated understanding

- [2023] **Multi-Grained Knowledge Retrieval for End-to-End Task-Oriented Dialog.** *Wan et al Arxiv.* [[paper]](https://arxiv.org/pdf/2305.10149.pdf) [[code]](https://github.com/18907305772/MAKER)
- [2022] **DialoKG: Knowledge-Structure Aware Task-Oriented Dialogue Generation.** *Rony et al NAACL.* [[paper]](https://aclanthology.org/2022.findings-naacl.195.pdf) [[code]](https://github.com/rashad101/DialoKG )
- [2022] **GraphMemDialog: Optimizing End-to-End Task-Oriented Dialog Systems Using Graph Memory Networks.** *Wu et al AAAI.* [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21403/21152)
- [2020] **Contextualize Knowledge Bases with Transformer for End-to-end Task-Oriented Dialogue Systems .** *Gou et al EMNLP.* [[paper]](https://aclanthology.org/2021.emnlp-main.353.pdf)
- [2020] **FG2SEQ: EFFECTIVELY ENCODING KNOWLEDGE FOR END-TO-END TASK-ORIENTED DIALOG .** *He et al ICASSP .* [[paper]](https://ieeexplore.ieee.org/document/9053667) [[code]](https://github.com/scoyer/FG2Seq)
- [2020] **GraphDialog: Integrating Graph Knowledge into End-to-End Task-Oriented Dialogue Systems .** *Yang et al EMNLP.* [[paper]](https://aclanthology.org/2020.emnlp-main.147.pdf) [[code]](https://github.com/shiquanyang/GraphDialog)

## Benchmark

## Modularly EToD on MultiWOZ2.0 and MultiWOZ2.1

### MultiWOZ2.0

| Model          | Year | Type       | Inform | Success | BLEU | Combined |
|----------------|------|------------|--------|---------|------|----------|
| SFN+RL         | 2019 | Modularly  | 73.8   | 58.6    | 16.9 | 83       |
| DAMD           | 2019 | Modularly  | 76.3   | 60.4    | 16.6 | 85       |
| LAVA           | 2020 | Modularly  | 91.8   | 81.8    | 12   | 98.8     |
| SimpleToD      | 2020 | Pretrain   | 84.4   | 70.1    | 15   | 92.3     |
| MinTL-BART     | 2020 | Pretrain   | 84.9   | 74.9    | 17.9 | 97.8     |
| UBAR           | 2020 | Pretrain   | 95.4   | 80.7    | 17   | 105.1    |
| AuGPT          | 2021 | Pretrain   | 83.1   | 70.1    | 17.2 | 93.8     |
| SOLOIST        | 2021 | Pretrain   | 85.5   | 72.9    | 16.5 | 95.7     |
| MTToD          | 2021 | Pretrain   | 91     | 82.6    | 21.6 | 108.3    |
| PPTOD          | 2021 | Pretrain   | 89.2   | 79.4    | 18.6 | 102.9    |
| SimpleToD-ACN  | 2022 | Pretrain   | 85.8   | 72.1    | 15.5 | 94.5     |
| BORT           | 2022 | Pretrain   | 93.8   | 85.8    | 18.5 | 108.3    |
| GALAXY         | 2022 | Pretrain   | 94.4   | 85.3    | 20     | 110.3    |

### MultiWOZ2.1

| Model          | Time | Type       | Inform | Success | BLEU | Combined |
|----------------|------|------------|--------|---------|------|----------|
| MD-Sequicity   | 2018 | Modularly  | 66.4   | 45.3    | 15.5 | 71.4     |
| SFN+RL         | 2019 | Modularly  | 73.8   | 58.6    | 16.9 | 83       |
| DAMD           | 2019 | Modularly  | 76.4   | 60.4    | 16.6 | 85       |
| UniConv        | 2020 | Modularly  | 72.6   | 62.9    | 19.8 | 87.6     |
| LABES-S2S      | 2020 | Modularly  | 78.1   | 67.1    | 18.1 | 90.7     |
| SimpleToD      | 2020 | Pretrain   | 85     | 70.5    | 15.2 | 93       |
| UBAR           | 2020 | Pretrain   | 95.7   | 81.8    | 16.5 | 105.3    |
| AuGPT          | 2021 | Pretrain   | 83.5   | 67.3    | 17.2 | 92.6     |
| SOLOIST        | 2021 | Pretrain   | 85.5   | 72.9    | 16.5 | 95.7     |
| MTToD          | 2022 | Pretrain   | 91     | 82.1    | 21   | 107.5    |
| PPTOD          | 2021 | Pretrain   | 87.1   | 79.1    | 19.2 | 102.3    |
| GALAXY         | 2022 | Pretrain   | 95.3   | 86.2    | 20   | 110.8    |
| SPACE3         | 2022 | Pretrain   | 95.3   | 88      | 19.3 | 111      |
| BORT           | 2022 | Pretrain   | 93.8   | 85.8    | 18.5 | 108.3    |

## Modularly EToD on CamRest6762

| Model          | Time | Match | Success | BLEU | Combined |
|----------------|------|-------|---------|------|----------|
| NDM            | 2016 | 90.4  | 83.2    | 21.2 | 108      |
| MD-Sequicity   | 2018 | 92.7  | 85.4    | 25.3 | 114.35   |
| FSDM           | 2019 | 93.5  | 86.2    | 25.8 | 115.65   |
| MOSS           | 2019 | 95.1  | 86      | 25.9 | 116.45   |
| LABES-S2S      | 2020 | 96.4  | 82.3    | 25.6 | 114.95   |
| ARDM           | 2021 | -     | 86.2    | 25.4 | -        |
| SOLOIST        | 2020 | -     | 87.1    | 25.5 | -        |
| BORT           | 2022 | -     | 89.7    | 25.9 | -        |
| SPACE3         | 2022 | 97.7  | 88.2    | 23.7 | 116.67   |

## Fully EToD on SMD

| Model          | Time | KB Type   | BLEU | Ent_F1 | Sch_F1 | Wea_F1 | Nav_F1 |
|----------------|------|--------|------|--------|--------|--------|--------|
| KVRet          | 2017 | Triplet| 13.2 | 48     | 62.9   | 53.3   | 44.5   |
| Mem2Seq        | 2018 | Triplet| 12.6 | 33.4   | 49.3   | 32.8   | 20     |
| GLMP           | 2019 | Triplet| 14.8 | 60     | 69.6   | 62.6   | 53     |
| BossNet        | 2019 | Triplet| 8.3  | 35.9   | 50.2   | 34.5   | 21.6   |
| KB-Transformer | 2019 | Triplet| 13.9 | 37.1   | 51.2   | 48.2   | 23.3   |
| DDMN           | 2020 | Triplet| 17.7 | 55.6   | 65     | 58.7   | 47.2   |
| DFNet          | 2020 | Triplet| 14.4 | 62.7   | 73.1   | 57.6   | 57.9   |
| DSR            | 2018 | Row    | 12.7 | 51.9   | 52.1   | 50.4   | 52     |
| MLM            | 2018 | Row    | 15.6 | 55.5   | 67.4   | 54.8   | 45.1   |
| KB-retriever   | 2019 | Row    | 13.9 | 53.7   | 55.6   | 52.2   | 54.5   |
| HM2Seq         | 2022 | Row    | 14.6 | 63.1   | 73.9   | 64.4   | 56.2   |
| Fg2Seq         | 2020 | Graph  | 16.8 | 61.1   | 73.3   | 57.4   | 56.1   |
| GraphDialog    | 2020 | Graph  | 13.7 | 60.7   | 72.8   | 55.2   | 54.2   |
| GraphMemDialog | 2021 | Graph  | 18.8 | 64.5   | 75.9   | 62.3   | 56.3   |
| GPT2KE         | 2021 | Graph  | 17.4 | 59.8   | 72.6   | 57.7   | 53.5   |
| COMET          | 2021 | Graph  | 17.3 | 63.6   | 77.6   | 58.3   | 56     |
| DialoKG        | 2022 | Graph  | 20   | 65.9   | x      | x      | x      |
| MAKER          | 2023 | Graph  | 25.91| 71.30  | x      | x      | x      |

## Fully EToD on MultiWOZ2.1

| Model          | Time | Type   | BLEU | Ent_F1 | Res_F1 | Att_F1 | Hot_F1 |
|----------------|------|--------|------|--------|--------|--------|--------|
| Mem2Seq        | 2018 | Triplet| 6.6  | 21.6   | 22.4   | 22     | 21     |
| MLM            | 2018 | Row    | 9.2  | 27.8   | 29.8   | 27.4   | 25.2   |
| DSR            | 2018 | Row    | 9.1  | 30     | 33.4   | 28     | 27.1   |
| BossNet        | 2019 | Triplet| 5.7  | 25.3   | 26.2   | 24.8   | 23.4   |
| GLMP           | 2019 | Triplet| 6.9  | 32.4   | 38.4   | 24.4   | 28.1   |
| DDMN           | 2020 | Triplet| 12.4 | 31.4   | 30.6   | 32.9   | 30.6   |
| DFNet          | 2020 | Triplet| 9.4  | 35.1   | 40.9   | 28.1   | 30.6   |
| Fg2Seq         | 2020 | Graph  | 13.5 | 36     | 40.4   | 41.7   | 30.9   |
| GraphMemDialog | 2021 | Graph  | 14.9 | 40.2   | 42.8   | 48.8   | 36.4   |
| MAKER          | 2023 | Graph  | 18.77| 54.72  | x      | x      | x      |

## Citation
If you find this repository useful, please cite our paper:

```
@inproceedings{qin2023end,
  title = {End-to-end Task-oriented Dialogue: A Survey of Tasks, Methods, and Future Directions},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  author = {Qin, Libo and Pan, Wenbo and Chen, Qiguang and Liao, Lizi and Yu, Zhou and Zhang, Yue and Che, Wanxiang and Li, Min},
  year = {2023},
  publisher = {Association for Computational Linguistics},
  doi = {To be determined},
  url = {To be determined},
  urldate = {2023-11-14}
}
```

## Project Maintainers & Contributors

Wenbo Pan (pixelwenbo@gmail.com), Qiguang Chen (qgchen@ir.hit.edu.cn) and Libo Qin (lbqin@csu.edu.cn)


<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WooooDyy/LLM-Agent-Paper-List&type=Date)](https://star-history.com/#WooooDyy/LLM-Agent-Paper-List&Date) -->

