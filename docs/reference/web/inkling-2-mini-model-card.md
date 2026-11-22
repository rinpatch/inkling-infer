# Inkling 2 Mini Model Card

## 1. General information

### Model name

Inkling 2 Mini

### Legal name of the model provider

Thinking Machines Lab, Inc.

### Date of release

November 20, 2026

### License

Apache 2.0

### Intended uses

Inkling 2 Mini is a general-purpose multimodal model that accepts text and image inputs and generates text outputs. It is intended for use in English and other languages, and across multiple coding languages. The model is designed to be used by developers building AI-powered applications, including agentic and tool-use systems, coding assistants, chatbots, and retrieval-augmented generation systems, and is suitable for general-purpose conversational use, instruction-following, and other natural language and visual tasks. It is released with open weights to support research, fine-tuning and integration into third-party products by downstream developers.

[Model Acceptable Use Policy](/model-acceptable-use-policy/)

### Languages

English, general multilingual support

## 2. Model properties

### Model type

Multimodal autoregressive looped transformer

### Architecture type

A 24-layer decoder-only transformer with sparse Mixture-of-Experts (MoE) feed-forward layers. The middle 12 layers run three times with shared weights, giving 48 layer applications per token from 24 layers of weights. In each layer, each token is routed to 8 of 128 experts. Attention is a hybrid of local and global layers. The model is natively multimodal – images are encoded via a vision transformer – with image and text inputs projected into a shared hidden space and processed jointly by the decoder.

### Parameters

26.4B, 3.4B active

Note: each extra pass through the looped layers adds about 1.0B effective active parameters. Since the model runs with three passes, there are 5.4B effective active parameters per token.

### Numerics support

NVFP4 and MXFP8

### Input modalities

- Text: UTF-8 encoded text
- Image: Any pixel-based image input. For optimal performance, each image dimension should be between 40px to 4096px.

Inkling 2 Mini supports a context window of up to 1M tokens.

### Output modalities

Text

## 3. Methods of distribution

The weights are available for download through Hugging Face [here](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

Developers can also access the model using the Tinker fine-tuning platform, which provides API access.

The model is also available via API access through third party inference providers.

### API access to the model

- For accessing our model via Tinker, you can get started by referring to the Tinker Cookbook [here](https://github.com/thinking-machines-lab/tinker-cookbook), and installing the tml-renderers package [here](https://pypi.org/project/tml-renderers/).
- Tinker playground also provides inference-only access to the model [here](https://tinker.thinkingmachines.ai/playground).
- For accessing our model via third party inference providers, you can refer to the documentation from our inference provider partners.

### Running the open weights model on your own system

#### Required hardware

The model is distributed as a single checkpoint of about 20GB, in mixed NVFP4 and MXFP8 precisions.

Given its small size, it is possible to run this checkpoint on local AI compute for inference. We support:

- 1x NVIDIA DGX Spark
- Apple silicon devices with 32GB or more memory (MacBooks, Mac minis and Studios)
- Other GPUs and platforms are possible, with 32GB or more of VRAM recommended

#### Required software

Running the model directly on your own hardware requires an inference deployment framework:

- For Apple silicon: [Ollama](https://ollama.com/library/inkling-2-mini), Exo Labs
- For NVIDIA: [SGLang](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini), [vLLM](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini), Unsloth, [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/recipes/inkling-2-mini), Tokenspeed, along with all of their respective dependency libraries.
- Quantized GGUF versions are available via [Unsloth](https://unsloth.ai/docs/inkling-2-mini) and llama.cpp for local inference
- For RL fine-tuning: [Miles](https://miles.radixark.com/docs/models/thinkingmachines/inkling-2-mini), [SkyRL](https://docs.skyrl.ai/docs/models/inkling2), [NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/vlm/thinkingmachines/inkling-2-mini.mdx)

## 4. Training

### Data types

Training data includes a broad variety of content types, including text, images, video.

### Data provenance

Training data for the model was drawn from publicly available sources, acquired from third-parties, generated internally, or synthetically generated or augmented. Publicly available data includes content from the public internet and publicly accessible repositories.

### Curation methodologies

Data curation includes cleaning, processing, and modifying datasets. These processing steps, which vary by data type, may include deduplication and filtering to remove junk or other low-quality data, or to advance safety or other objectives.

## 5. Evaluations

|  | Looped TransformersLooped Transformers |  |
|----|----|----|
|  | Inkling 2 Mini | Dory 1.0 4B |
| Model Info |  |  |
| Params (B)activated / total | 5.4/26.4 | –/4.0 |
| \# Passes | 3 | 4 |
| Third Party Index |  |  |
| AA Index (v5.0) | 20.1 (Best reported score in this row) | – |
| Vals Index (v2.2) | 46.5% (Best reported score in this row) | – |
| Coding and Autoresearch |  |  |
| FrontierCode v2.0stirrup v0.2.1 | 15.1% (Best reported score in this row) | – |
| DeepSWE v1.1mini-swe-agent v2.4.6 | 48.2% (Best reported score in this row) | 0.4% |
| Terminal Bench v4.0mini-swe-agent v2.4.6 | 5.6% (Best reported score in this row) | 0.0% |
| Terminal Bench v2.1Inkling 2 Mini in pi v0.83.0 | 71.6% | 22.2% |
| SWEBench Pro v2 (Hard)mini-swe-agent v2.4.6 | 77.9% (Best reported score in this row) | 13.2% |
| SWEBench MultilingualInkling 2 Mini with bash only | 73.7% (Second-best reported score in this row) | 43.0% |
| SWEBench Verifiedbash only | 79.2% (Second-best reported score in this row) | 61.8% |
| SWEBench Multimodal v2mini-swe-agent (vals) | 37.9% (Best reported score in this row) | – |
| Vibe Code Bench v1.1openhands (vals) | 77.9% (Best reported score in this row) | – |
| Code Migration v1.1pi (vals) | 23.2% (Best reported score in this row) | – |
| Terminal Bench Science v0.1mini-swe-agent v2.4.6 | 1.4% (Best reported score in this row) | 0.0% |
| PMPP-Hard (AUC, 0–10h)pi v0.83.0 | 455.82 (Best reported score in this row) | – |
| Agentic Search, Tool Use, and Knowledge Work |  |  |
| BrowseCompInkling 2 Mini with compaction | 79.0% (Best reported score in this row) | 38.9% |
| DeepSearchQAInkling 2 Mini with compaction | 85.3% (Best reported score in this row) | 66.0% |
| MCP AtlasPublic | 82.7% (Second-best reported score in this row) | 54.2% |
| AA-Briefcase v1.1 | 1263 (Second-best reported score in this row) | – |
| GDPVal-AA v2.1 | 1319 (Second-best reported score in this row) | – |
| JobBench | 49.6% (Best reported score in this row) | – |
| AutomationBench-AA v1.0.6 | 41.8% | – |
| EnterpriseOps-Gym | 42.0% (Second-best reported score in this row) | 26.0% |
| Finance Agent v2.1 | 56.5% (Best reported score in this row) | – |
| Excel Modeling Benchmark v1.1 | 66.0% (Best reported score in this row) | – |
| Tax Agent Bench | 14.1% (Second-best reported score in this row) | – |
| AA-AnalystAgent | 46.3% (Best reported score in this row) | – |
| Diligence BenchBest harness | 48.6% (Best reported score in this row) | – |
| Vision Interpretation and Creation |  |  |
| GDP.pdf | 16.8% (Best reported score in this row) | – |
| PerceptionBench | 57.5% (Best reported score in this row) | – |
| MMMU ProStandard 10 | 74.9% (Second-best reported score in this row) | – |
| OmniDocBench v1.6 | 92.8% (Best reported score in this row) | – |
| BabyVisionoriginal / with Python | 53.4/88.8% (Best reported score in this row) | – |
| ScreenSpot Prooriginal / with Python | 76.5/87.7% (Best reported score in this row) | – |
| Charxiv RQoriginal / with Python | 80.6/88.3% (Second-best reported score in this row) | – |
| Chartographyoriginal / with Python | 22.1/72.5% (Best reported score in this row) | – |
| ZeroBench (main)original / with Python | 17.0/34.0% (Best reported score in this row) | – |
| GroundingME | 55.5% (Best reported score in this row) | – |
| SAGE v1.1 | 85.9% (Best reported score in this row) | – |
| MathVision | 89.0% (Second-best reported score in this row) | – |
| MuirBench | 77.5% (Second-best reported score in this row) | – |
| BVB | 68.4% (Second-best reported score in this row) | – |
| Code2Video-AA | 870 (Second-best reported score in this row) | – |
| Reasoning, Knowledge, and Epistemics |  |  |
| HLE-Diamond | 15.2% (Second-best reported score in this row) | 6.2% |
| HLE (text only) | 30.7% | 10.4% |
| CritPt v2Full / Hard | 31.4/17.0% (Second-best reported score in this row) | – |
| GPQA Diamond | 86.6% | 72.2% |
| Apex 2025 | 34.4% (Best reported score in this row) | 1.0% |
| ARC-AGI-1 | 84.5% (Second-best reported score in this row) | – |
| ARC-AGI-2 | 49.3% (Best reported score in this row) | – |
| MysteryMechanism | 23.0% (Best reported score in this row) | – |
| ProofBench v1.1 | 18.0% (Best reported score in this row) | – |
| AA-LCR v1.1 | 80.0% | 77.0% |
| LongBench v2 | 66.9% (Best reported score in this row) | 41.5% |
| GraphWalks BFS256K | 69.7% (Best reported score in this row) | – |
| MRCR v28-needle 256K–512K | 46.4% (Best reported score in this row) | – |
| AA-Omniscience v1.1score | -8.6 | -7.7 |
| AA-Omniscience v1.1non-hallucination rate | 63.4% | – |
| ComplexConstraints v1.1openhands v1.45.0 | 33.4% | 17.9% |
| Safety and Alignment |  |  |
| FORTRESS (adversarial)Public | 89.5% | 91.9% (Best reported score in this row) |
| FORTRESS (benign)Public | 96.4% | 79.2% |
| StrongREJECT (none) | 97.8% | 99.0% |
| Misbehavior (Internal) ↓ | 9.8% (Best reported score in this row) | – |
| Covert misbehavior (Internal) ↓ | 8.3% (Best reported score in this row) | – |
| Destructive actions (Internal) ↓ | 10.2% (Best reported score in this row) | – |
| HoneyBench v0.2 ↓ | 56.7% | – |

Best reported score in each row (lowest where lower is better); equal displayed scores tieSecond-best reported score in each row; none when first place is tied

## 6. Safety

We conducted safety evaluations ahead of release, spanning both everyday human-AI interaction and dangerous-capability testing. Because Inkling 2 Mini is multimodal, we paid attention to whether safety behavior held consistently across text and image inputs. We applied mitigations to reduce risks before release, including an extra phase of alignment training using on-policy distillation, similarly to Inkling 2 Small.

For everyday use, we evaluated sycophancy and how the model handles requests involving suicide, self-harm and eating disorders, including whether it encourages harm or affirms false beliefs. We also assessed whether the model refuses genuinely harmful requests without over-refusing benign ones. For CBRN and cyber, we assessed knowledge and procedural uplift through internal evaluations and refusal-suppressed variants intended to estimate latent capability with safeguards removed. For loss of control, we evaluated agentic capability, strategic deception, and sabotage potential, benchmarked against public frontier models, and found the model materially below frontier capabilities.

Inkling 2 Mini is the first looped transformer we have released, so we also studied how looping affects monitorability: whether we can still oversee the model by reading its chain of thought. Looping lets the model do more computation without reasoning in visible tokens. Inkling 2 Mini has very little controllability over its chain of thought, and was unable to perform covert side tasks in realistic agentic environments. Further experiments on the effect of looping on monitorability are described in our [blog post](/news/inkling-2-mini/).

Across all areas, we concluded that Inkling 2 Mini does not give material uplift beyond what is already available from open-weights models.

## 7. Bias, risks and limitations

Inkling 2 Mini may exhibit general limitations common to foundation models, including hallucination (generating plausible but factually incorrect or unsupported content), occasional failures to follow instructions precisely, and degraded performance in long multi-turn conversations. As with other models trained on web-derived and synthetic data, it may reflect biases present in its training data, including demographic, cultural or linguistic biases, and may perform unevenly across languages, dialects or subject domains that were less represented during training.

Inkling 2 Mini’s knowledge is limited to information available as of its training cutoff, and it may not reflect events or changes that occurred afterward.

Inkling 2 Mini is a small model built to run on local hardware, and some limitations are specific to it:

- Knowledge and factual accuracy. It is less capable than Inkling 2 Small across most of our evaluations, with the largest gaps on factual recall and knowledge-heavy questions. It is more likely than larger models to state incorrect information with confidence.
- Repetitive reasoning. In internal testing the model sometimes became stuck in long reasoning loops, or repeatedly re-checked its own work without making progress, particularly when debugging code. We recommend setting limits on output tokens and agent turns.
- Agentic reliability. When using tools, it is less reliable than larger models at following detailed instructions and project conventions. It may take destructive actions, such as deleting or overwriting files, without confirmation, and may not report the outcome of its work accurately.
- Languages other than English. Output quality can be lower outside English. In testing in Chinese we observed clipped phrasing and mixed languages.
- Sensitive topics. The model can respond with more caution than the situation calls for when a conversation touches on mental health.
- Reasoning and compute. Setting reasoning effort to its minimum does not fully disable reasoning. Because the middle layers run three times, compute per token is higher than the active-parameter count alone suggests.
- Local performance. Support for looped models in local runtimes is new, and speed varies by runtime and hardware.

We recommend that downstream developers and deployers:

- Apply appropriate human oversight and review for outputs used in high-stakes or safety-critical contexts, rather than relying on Inkling 2 Mini’s outputs without verification.
- Run agentic workloads in a sandbox, and require confirmation before destructive or irreversible actions.
- Conduct their own evaluation of Inkling 2 Mini’s performance, safety and fairness for their specific use case, language and population prior to deployment, particularly for applications involving vulnerable groups.
- Implement additional safeguards, such as content filtering, rate limiting and monitoring, at the application layer, especially for open deployment contexts where Inkling 2 Mini’s built-in mitigations may not be sufficient on their own.
- Avoid deploying Inkling 2 Mini in domains such as medical, legal or safety-critical decision-making without additional fine-tuning, domain-specific validation and human oversight.

## 8. Legal

[Training Data Documentation](/training-data-documentation/)
