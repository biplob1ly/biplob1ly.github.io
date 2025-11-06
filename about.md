---
layout: page
title: About Me
---

I work as a reseach engineer at Fujitsu Research of America (FRA). I earned a Ph.D. from the [Department of Computer Science and Engineering, The Ohio State University (OSU)](https://cse.osu.edu/) supervised by Prof. [Rajiv Ramnath](https://web.cse.ohio-state.edu/~ramnath.6/). My research focuses on the application of in the application of machine learning, especially in **Natural Language Processing (NLP)**. Before joining OSU, I obtained my B.Sc. in CSE from [Bangladesh Universiy of Engineering and Technology (BUET)](https://www.buet.ac.bd/). I also worked at [Kona Software Lab Ltd](https://konasl.com/) at the Dhaka site for 2 years. 

I have experience in employing neural **information retrieval (IR)** and **reinforcement learning (RL)** techniques to develop **retrieval-guided response generation (NLG)** models for conversational agents. At FRA, I have designed and developed a multi-agent system (RAG, Coding) for an AI chatbot to analyze tabular and textual data [Python, LangChain, LangGraph, Ollama, Streamlit, Chroma].

Publications
------------

- [Efficient and Interpretable Information Retrieval for Product Question Answering with Heterogeneous Data](https://aclanthology.org/2024.ecnlp-1.3/), **Biplob Biswas**, Rajiv Ramnath. _In Proceedings of the Seventh Workshop on e-Commerce and NLP @ LREC-COLING_, May 2024.

- [Improving Response Automation for Customer Communication through Document Categorization and Information Retrieval](http://rave.ohiolink.edu/etdc/view?acc_num=osu170064805272096), **Biplob Biswas**. Doctoral dissertation, _The Ohio State University_, 2023.

- [Retrieval Based Response Letter Generation For a Customer Care Setting](https://aclanthology.org/2022.naacl-industry.20/), **Biplob Biswas**, Renhao Cui, Rajiv Ramnath. _In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL): Human Language Technologies: Industry Track_, July 2022.

- [TransICD: Transformer Based Code-wise Attention Model for Explainable ICD Coding](https://doi.org/10.1007/978-3-030-77211-6_56), **Biplob Biswas**, Hoang Pham, Ping Zhang. _In International Conference on Artificial Intelligence in Medicine (AIME)_, Jun 2021.

- [Towards Simulating Non-lane Based Heterogeneous Road Traffic of Less Developed Countries](https://easychair.org/publications/open/tvN9), Q. M. Alam, B. Sarker, **Biplob Biswas**, K. H. Zubaer, T. Toha, N. Reza and ABM A. Al Islam. _In ICT4S_, May 2018.

Projects
--------

- Designed and developed a **multi-agent system** (RAG, Coding) for an AI chatbot to analyze tabular and textual data. _\[Python, LangChain, LangGraph, Ollama, Streamlit, Chroma\]_

- Designed a **sequence generation model** for predictive process monitoring.(Python, PyTorch)

- Developed an ML-driven web tool for process data analysis and visualization.(Python, Streamlit)

- Designed and developed **a knowledge-grounded response generation framework** to automatically answer user queries in a customer care setting (Real-world data) and deployed it as an API service. _\[AWS, Docker, Python, PyTorch, HuggingFace\]_
	* Employed neural **information retrieval (IR)** techniques to collect knowledge from historical unstructured data and blended that information into generated responses through a fine-tuned GPT-2 model.
	* Formulated **a novel ranking mechanism** for better response selection from multiple generations.
	* The model generates **12% more informative and 15.3% more semantically similar responses** than that of vanilla GPT-2 (contemporary state-of-the-art). 

- Developed **a paraphrasing model and corresponding web service** to introduce diversity in the business response template. _\[AWS, Docker, Python, Python\]_

- Designed and developed **a deep-learning model - TransICD** ([Code](https://github.com/biplob1ly/TransICD)) to predict ICD codes from a patient discharge summary. _\[Python, PyTorch\]_
	* The model achieves stable **micro-AUC and micro-F1 scores of 0.92 and 0.64 respectively** with 1.1% and 2.5% increases over the corresponding metrics of LEAM (contemporary state-of-the-art). 

- Developed a tool to extract and analyze pre-installed applications in Android Firmware. _\[Python, MySQL\]_

- Developed the front-end of [NexusPay](https://play.google.com/store/apps/details?id=com.dbbl.nexus.pay) **a leading Android payment application (1M+ Downloads)** of Bangladesh. _\[Android, Java, SQLite\]_
	* Utilized **REST APIs** for backend communication and implemented various transaction interfaces including NFC, QR-code, and online payment.

- Developed **a Nexgo POS application** to facilitate the transaction from an Android payment application. _\[C\]_