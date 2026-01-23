<div align="center">

# Eduardo Marinho 
[**GitHub**](https://github.com/enfmarinho) | [**LinkedIn**](https://linkedin.com/in/enfmarinho) | [**Email**](mailto:contact@eduardomarinho.dev)

**Low-Level Systems & High-Performance Software Engineer**

**BS in Computer Science @ Federal University of Rio Grande do Norte** (Expected Dec 2027)

</div>

##### About
*I build performance-critical, real-time software systems with a focus on deterministic execution, memory efficiency, and empirical validation. My work spans low-level C/C++ systems, Rust-based tooling, and data-driven experimentation pipelines.*

This portfolio highlights projects where I owned **end-to-end system design**, from architecture and low-level optimization to validation and tooling.

---

## Impact Highlights
* **Real-Time Systems**: Developed a high-performance chess engine where **evaluation latency directly determines system behavior**, achieving a ~105% throughput increase through SIMD and memory-layout optimization.
* **Production Rust**: Published a Rust engine on crates.io with **9,200+ downloads**, demonstrating production-grade software and project ownership.
* **Empirical Engineering**: Every performance-critical change validated using **statistical testing, test suits, benchmarks, and regression detection**.

---

## Technical Case Studies

> These case studies emphasize systems design under constraints where latency, determinism, and failure modes directly affect behavior.

### Systems & Performance
---
#### Minke | High-Performance Chess Engine
*C++ (core engine) | Rust (tooling) | SIMD-vectorized x86-64 and Neon kernels | concurrency*
> **[Full Case Study](casestudies/minke.md)**
>
>**[Source Code](https://github.com/enfmarinho/Minke)**

Performance-critical system where microsecond-level latency drives overall behavior. While the domain is chess, the engineering challenges closely resemble control and automation systems:
* Tight inner loops executed millions of times per second
* Strong sensitivity to cache behavior and memory layout
* Deterministic execution under real-time constraints
* Continuous validation to prevent regressions

**Highlights**
* Designed SIMD-accelerated kernels for neural network inference (AVX2, AVX512, NEON), doubling evaluation throughput.
* Engineered cache-aligned data structures and bitboard-based state representation, reducing memory stalls and branch misprediction.
* Built a Rust-based orchestration CLI to manage training, benchmarking, and validation as a reproducible pipeline.
* Validated search behavior and performance using deterministic benchmarks and distributed testing.

---

#### Tgames | Asynchronous Rust TUI Engine
*Rust | Concurrency | Software Engineering and Design*
<!-- > **[Full Case Study](casestudies/minke.md)** -->
<!-- > -->
> **[Source Code](https://github.com/enfmarinho/tgames)**

Tgames is a terminal-based Rust engine structured around a deterministic update loop that separates input handling, state updates, and rendering. This design enables non-blocking input processing while preserving predictable execution order and state transitions under concurrent workloads.

**Key engineering properties:**
* Deterministic update and render ordering
* Explicit state model with clear ownership boundaries
* Concurrency-safe, non-blocking input handling
* Trait-based abstractions enabling extensibility without runtime modification

---

#### Compiler Prototype (Comp) 
*C++ | Flex | Bison*

*A compiler designed as a multi-stage lowering pipeline from a custom language into C Three Address Code (TAC).*
* **Language Design:** Developed a linear IR to simplify expression lowering and support backend code generation.
* **Semantic Analysis:** Implemented passes for **symbol table construction** and **static type checking** to ensure program correctness prior to lowering.
* **Code Generation:** Engineered a backend that translates **Three-Address Code (TAC)** into portable C for integration with GCC/Clang.

#### NoC Simulator 
*C++ | SystemC*

- Built a cycle-accurate 2D mesh Network-on-Chip simulator with deadlock-free routing, arbitration, and flow control.
- Implemented virtual-cut-through routers with round-robin arbitration, crossbar switching, and FIFO buffering.
- Verified correctness and performance using automated testbenches and VCD waveform analysis.

---

#### [Ducta](https://github.com/enfmarinho/ducta) | High-Throughput HTTP Server (Work-in-Progress)
*Rust | Linux (io_uring/epoll) | Zero-copy | Event-driven Design*

Early-stage project exploring high-throughput Linux networking in Rust. Focused on:
- Asynchronous I/O with io_uring and epoll
- Zero-copy request parsing for minimal latency
- Custom router for high concurrency

---

### Computer Vision & Applied Research

---

### Pasture Biomass 
*PyTorch | TorchVision | Pandas | sklearn*

Automated computer vision pipeline to predict biomass components from ground-level imagery. The system utilizes a state-of-the-art Vision Transformer (ViT) backbone to achieve high-precision estimation in agricultural environments.

* **Backbone Architecture**: pre-trained DINOv2 (ViT−S/14) as the primary feature extractor.
* **Multitask Learning**: Designed a shared backbone architecture with 4 auxiliary heads to automate simultaneous biomass estimation tasks.
* **Performance Optimization**: Improved model accuracy by 11% through the implementation of ensemble prediction and Test Time Augmentation (TTA).
* **Robust Validation**: Leveraged multitask learning with auxiliary metadata and validated results via 5-fold cross-validation to ensure generalization.

#### Cardiovascular Diagnostic Research | Image Processing
* **Pipeline Design**: Engineered data preparation and denoising pipelines for medical imaging datasets to optimize feature extraction for deep learning models.
* **Research Focus**: Collaborated on multi-disciplinary teams to bridge the gap between clinical requirements and automated medical diagnostics.

---

### Backend & Data Pipelines

#### EchoTyper | Multi-Stage Data Orchestration
*Spring Boot | PostgreSQL | Design patterns and Software Architecture*

* **Pipeline Design**: Architected a multi-stage backend to orchestrate data flow between external ML services and persistent storage, ensuring high availability and system modularity.
* **Service Abstraction**: Implemented the Design Patterns to decouple core logic from external API providers, a design choice aimed at reducing vendor lock-in and reduce code duplication.
* **Database:** Data pipeline that persists meetings transcription and summarization metadata in Postgres

---

## Technical Toolbox

### Languages
* **C / C++**: Real-time systems, memory layout optimization and tooling
* **Rust**: Memory-safe systems programming and automation tooling (familiar with memory safety and ownership models).
* **Python**: Data analysis, ML pipelines, and rapid hardware/software prototyping.

---

### Computer Architecture & Low-Level Systems & Embedded Systems
* **Hardware Awareness**: Deep understanding of CPU cache hierarchies (L1/L2/L3), memory locality, and Instruction-Level Parallelism.
* **Operating Systems**: Experience with Linux/Unix environments, process management, and concurrency programming.
* **SIMD intrinsics**: Implementation of hand-tuned kernels using AVX2, AVX-512, and NEON intrinsics to maximize throughput.
* **Tooling**: Build systems (CMake/Make), debuggers (GDB/LLDB), memory profilers (Valgrind) and profilers (perf).
* **Control & Robotics**: Familiarity with feedback loops, state-space search, and trajectory optimization concepts.

---

### Software/Data Engineering & Infrastructure
* **Software Architecture**: structural and behavioral patterns to architect modular systems and hardware-abstraction layers, enabling clean separation between high-level logic and low-level driver implementations.
* **Data Pipelines**: Architecting reproducible pipelines in Python and Rust for capturing and analyzing complex process data.
* **Empirical Validation**: Statistical engine evaluation (SPRT), deterministic benchmarking, and regression detection to ensure system correctness.
* **CI/CD & Infrastructure**: Building automated testing pipelines with GitHub Actions, Docker, and distributed validation frameworks.

---

### Engineering Philosophy

I approach software as a system, not isolated code. My design decisions are guided by:

* Measurable performance, not assumptions
* Explicit trade-offs between latency, throughput, and complexity
* Tooling that enables fast iteration without sacrificing correctness

My goal is to deliver software that performs efficiently under real-world constraints and scales gracefully over time.

---
