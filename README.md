# Eduardo Marinho 
[**GitHub**](https://github.com/enfmarinho) | [**LinkedIn**](https://linkedin.com/in/enfmarinho) | [**Email**](mailto:contact@eduardomarinho.dev)

**Low-Level Systems & High-Performance Software Engineer**
*BS in Computer Science @ Federal University of Rio Grande do Norte (Expected July 2027)*

##### About
Systems programmer focused on high-throughput, low-latency architectures in C++ and Rust. I build deterministic, memory-efficient systems optimized at the hardware level, validated through rigorous statistical testing and profiling.

This portfolio highlights projects where I owned **end-to-end system design**, from architecture and low-level optimization to validation and tooling.

---

## Impact Highlights
* **Real-Time Systems**: Developed a high-performance chess engine where **evaluation latency directly determines system behavior**, achieving a ~105% throughput increase through SIMD and memory-layout optimization.
* **Production Rust**: Published a Rust engine on crates.io with **9,200+ downloads**, demonstrating production-grade software and project ownership.
* **Empirical Engineering**: Every performance-critical change validated using **statistical testing, test suites, benchmarks, and regression detection**.

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

Performance-critical C++ engine where microsecond-level latency and memory layout dictate heuristic depth. Designed around a deterministic search architecture, executing millions of state evaluations per second.
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
*Rust | Asynchronous I/O | State Machine | TUI*
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

## Technical Toolbox

### Languages
* **C / C++**: Real-time systems, memory layout optimization and tooling
* **Rust**: Memory-safe systems programming and automation tooling (familiar with memory safety and ownership models).
* **Python**: Data analysis, ML pipelines, and rapid hardware/software prototyping.

---

### Computer Architecture & Low-Level Systems
* **Hardware Awareness**: Deep understanding of CPU cache hierarchies (L1/L2/L3), memory locality, and Instruction-Level Parallelism.
* **Operating Systems**: Experience with Linux/Unix environments, process management, and concurrency programming.
* **SIMD intrinsics**: Implementation of hand-tuned kernels using AVX2, AVX-512, and NEON intrinsics to maximize throughput.
* **Tooling**: Build systems (CMake/Make), debuggers (GDB/LLDB), memory profilers (Valgrind) and profilers (perf).

---

### Software/Data Engineering & Infrastructure
* **Software Architecture**: structural and behavioral patterns to architect modular systems and hardware-abstraction layers, enabling clean separation between high-level logic and low-level driver implementations.
* **Automated Pipelines**: Architecting reproducible Rust and Python pipelines for distributed engine benchmarking, model training, and continuous integration.
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
