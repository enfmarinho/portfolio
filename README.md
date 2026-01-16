<div align="center">

# Eduardo Marinho 

**Systems & High-Performance Software Developer**

**BS in Computer Science @ Federal University of Rio Grande do Norte** (Expected Dec 2027)

*Focused on high-performance systems, memory-efficient design, and data-driven engineering*

[**GitHub**](https://github.com/enfmarinho) | [**LinkedIn**](https://linkedin.com/in/enfmarinho) | [**Email**](mailto:contact@eduardomarinho.dev)

</div>

---

## 🚀 Impact Highlights
* **Performance:** Achieved a **~105% increase in engine throughput** through SIMD-accelerated NNUE inference compared to baseline evaluation
* **Rank:** Developed the **2nd highest-ranked Brazilian chess engine** on the Computer Chess Rating List
* **Scale:** Published a Rust mini-games emulator with **9,200+ downloads** on crates.io

---

## Technical Case Studies
> High-performance and systems engineering projects demonstrating concurrency, memory optimization, and low-level architecture.
### 🛠️ Systems & Performance

#### [Minke](casestudies/minke.md) | High-Performance Chess Engine
*A NNUE-based C++ chess engine focused on search performance, memory locality and accurate position evaluation, ranked #2 on the CCRL.*
* **Performance Engineering:** Integrated NNUE evaluation and built **SIMD-accelerated kernels** for inference, resulting in a **~105% increase in Nodes Per Second (NPS)**.
* **Memory Optimization:** Utilized **bitboards** and **cache-efficient layouts** to achieve a **~27% speedup** in negamax search.
* **Automation & Validation:** Orchestrated a distributed testing framework and CI/CD workflows to continuously validate engine correctness and performance regressions; developed a custom Rust CLI to automate NN training and monitoring.
* **Full Case Study:** [Read here](casestudies/minke.md) for detailed architecture, heuristics, training pipeline, and data-driven validation.

#### [Tgames](https://github.com/enfmarinho/tgames) | Asynchronous Rust UI Engine
*A high-performance terminal UI engine designed for modularity and speed.*
* **Concurrency:** Architected a state-driven engine enabling **asynchronous input** and rendering for a suite of 5 interactive games.
* **Production Standards:** Published on crates.io with **9,200+ downloads**, demonstrating real-world adoption and production-quality code.
* **Design:** Applied the **Template Method pattern** with Rust traits and enums to ensure long-term maintainability and easy extension.

#### Compiler Prototype (Comp) | C++, Flex, Bison
*A compiler designed as a multi-stage lowering pipeline from a custom language into C Three Address Code (TAC).*
* **Language Design:** Developed a linear IR to simplify expression lowering and support backend code generation.
* **Semantic Analysis:** Implemented passes for **symbol table construction** and **static type checking** to ensure program correctness prior to lowering.
* **Code Generation:** Engineered a backend that translates **Three-Address Code (TAC)** into portable C for integration with GCC/Clang.

### 💾 Backend & Data Pipelines

#### EchoTyper | Scalable Data Pipelines
*An AI-powered web application for automated transcription, summarization and organization of meetings.*
* **API Architecture:** Designed a RESTful backend using Spring Boot to orchestrate multi-stage transcription and summarization pipelines with external ML services (Google STT, Gemini).
* **Service Abstraction:** Implemented the **Strategy design pattern** to abstract LLM providers, creating a modular system that reduced code duplication and allows easy service switching.
* **Database:** Architected a data pipeline that persists meetings transcription and summarization metadata in Postgres, handling schema design for multi-stage LLM output
* **Optimization:** Reduced token usage by **12%** while improving output quality through fine-tuned summarization prompts.

---

## 🤖 Applied ML & Scientific Research
* **Pasture Biomass Prediction:** Leveraged Vision Transformers (DINOv2) and multitask learning with multiple regression heads to predict biomass components. Developed automated ML pipelines with ensemble prediction and test-time augmentation (TTA), improving model accuracy by ~11%.

* **Cardiovascular Research:** Collaborated on computer vision projects applying deep learning to diagnose disease, specifically developing data preparation pipelines to reduce noise and optimize feature extraction.

---

## 🧰 Technical Toolbox
* **Languages:** C/C++, Rust, Java, TypeScript, Python, Go, SQL.
* **Libraries & Frameworks:** PyTorch, TorchVision, scikit-learn, Spring Boot, Bullet ML, GoogleTest.
* **Developer Tools:** Git, GitHub, CI/CD, Docker, CMake, Make, GDB/LLDB, Valgrind, Google Cloud Platform.
* **Relevant Coursework:** Operating Systems, Software Engineering, Software Architecture, Computer Architecture, Databases, Concurrent Programming, Deep Learning.

---

### 📖 Technical Philosophy
I focus on building reliable, maintainable, and high-performance software systems. My work emphasizes memory efficiency, concurrency, and predictable performance, informed by both practical experience and study of foundational texts like Designing Data-Intensive Applications and Design Patterns. I approach every project as a learning opportunity to understand system trade-offs, from algorithm design to data flow and infrastructure. Whether optimizing a search engine in C++ or architecting asynchronous Rust applications, my goal is to deliver software that performs efficiently under real-world constraints and scales gracefully over time.

---
