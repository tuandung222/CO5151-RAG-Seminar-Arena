# Active Retrieval Augmented Generation

**Authors:** Zhengbao Jiang¹†, Frank F. Xu¹†, Luyu Gao¹†, Zhiqing Sun¹†, Qian Liu², Jane Dwivedi-Yu³, Yiming Yang¹, Jamie Callan¹, Graham Neubig¹

†Equal contribution / Lead contributors.

¹Language Technologies Institute, Carnegie Mellon University; ²Sea AI Lab; ³FAIR, Meta

**Correspondence:** `{zhengbaj, fangzhex, luyug, zhiqings, gneubig}@cs.cmu.edu`

**Conference:** Findings of EMNLP 2023 | **ArXiv ID:** [2305.06983](https://arxiv.org/abs/2305.06983) | **Code:** [https://github.com/jzbjyb/FLARE](https://github.com/jzbjyb/FLARE)


---

## Abstract

Despite the remarkable ability of large language models (LMs) to comprehend and generate language, they have a tendency to hallucinate and create factually inaccurate output. Augmenting LMs by retrieving information from external knowledge resources is one promising solution. Most existing retrieval augmented LMs employ a retrieve-and-generate setup that only retrieves information once based on the input. This is limiting, however, in more general scenarios involving generation of long texts, where continually gathering information throughout generation is essential. In this work, we provide a generalized view of active retrieval augmented generation, methods that actively decide when and what to retrieve across the course of the generation. We propose Forward-Looking Active REtrieval augmented generation (FLARE), a generic method which iteratively uses a prediction of the upcoming sentence to anticipate future content, which is then utilized as a query to retrieve relevant documents to regenerate the sentence if it contains low-confidence tokens. We test FLARE along with baselines comprehensively over 4 long-form knowledge-intensive generation tasks/datasets. FLARE achieves superior or competitive performance on all tasks, demonstrating the effectiveness of our method.[^1]

[^1]: Code and datasets are available at [https://github.com/jzbjyb/FLARE](https://github.com/jzbjyb/FLARE).


> [!NOTE] Key Concepts & Contributions
> - **Active Retrieval Augmented Generation Framework:** A generalized framework where retrieval is interleaved throughout the generation process, actively deciding *when* and *what* to retrieve instead of retrieving only once or at passive, rigid intervals.
> - **Forward-Looking Formulation (FLARE):** Anticipates future content needs by generating a temporary next sentence $\hat{\bm{s}}_t$ rather than querying with only past context.
> - **Confidence-Based Triggering (*When* to retrieve):** Leverages LM calibration; retrieval is triggered only when candidate sentence $\hat{\bm{s}}_t$ contains tokens with probability lower than confidence threshold $\theta$.
> - **Query Formulation Strategies (*What* to retrieve):**
>   - **$\text{FLARE}_{\text{instruct}}$:** Instruction-tuned prompting where LM emits explicit search commands `[Search(query)]` during decoding.
>   - **$\text{FLARE}_{\text{direct}}$ (Implicit):** Masks low-confidence tokens (below threshold $\beta$) in $\hat{\bm{s}}_t$ to avoid distracting the retriever.
>   - **$\text{FLARE}_{\text{direct}}$ (Explicit):** Zero-shot question generation targeting low-confidence spans using an auxiliary LM (`gpt-3.5-turbo`).
> - **Superior Empirical Performance:** Tested across 4 diverse long-form knowledge-intensive benchmarks (2WikiMultihopQA, StrategyQA, ASQA, and WikiAsp), consistently outperforming single-time retrieval and passive multi-time retrieval baselines.


---

## 1 Introduction

Generative language models (LMs) Brown et al. (2020); Ouyang et al. (2022); OpenAI (2023); Chowdhery et al. (2022); Zhang et al. (2022); Touvron et al. (2023); Zhao et al. (2023) have become a foundational component in natural language processing (NLP) systems with their remarkable abilities.
Although LMs have memorized some world knowledge during training Petroni et al. (2019); Roberts et al. (2020); Jiang et al. (2020), they still tend to hallucinate and create imaginary content Maynez et al. (2020); Zhou et al. (2021).
Augmenting LMs with retrieval components that look up relevant information from external knowledge resources is a promising direction to address hallucination Khandelwal et al. (2020); Izacard et al. (2022).


### Figure 1: Architecture Overview of FLARE

```mermaid
flowchart TD
    A["User Input x & Initial Retrieval D_x"] --> B["Generate Temporary Next Sentence s_hat_t = LM([x, y_<t])"]
    B --> C{"Check Token Probabilities in s_hat_t: Any token < theta?"}
    C -- "No (High Confidence)" --> D["Accept Sentence: y_t = s_hat_t"]
    C -- "Yes (Low Confidence)" --> E["Formulate Query q_t (Implicit Masking or Explicit Question Gen)"]
    E --> F["Retrieve External Documents D_q_t = ret(q_t)"]
    F --> G["Regenerate Next Sentence: s_t = LM([D_q_t, x, y_<t])"]
    G --> H["Accept Regenerated Sentence: y_t = s_t"]
    D --> I{"Reached End of Generation?"}
    H --> I
    I -- "No" --> B
    I -- "Yes" --> J["Final Output y = [s_1, s_2, ..., s_m]"]
```

> **Figure 1 Caption:** An illustration of forward-looking active retrieval augmented generation (FLARE). Starting with the user input $\bm{x}$ and initial retrieval results $\mathcal{D}_{\bm{x}}$, FLARE iteratively generates a temporary next sentence (shown in gray italic) and checks whether it contains low-probability tokens (indicated with underline). If so (step 2 and 3), the system retrieves relevant documents and regenerates the sentence.

> [!NOTE] Figure 1 Detailed Workflow Breakdown
> 1. **Initial Context:** The generation begins with user input $\bm{x}$ and optional initial retrieval results $\mathcal{D}_{\bm{x}}$.
> 2. **Step 1 (Confident Generation):** The LM predicts temporary candidate sentence $\hat{\bm{s}}_1$. Because all token generation probabilities meet or exceed threshold $\theta$, the model accepts $\hat{\bm{s}}_1$ directly as $\bm{s}_1$ without triggering retrieval.
> 3. **Step 2 (Uncertain Generation & Active Retrieval):** The LM predicts candidate sentence $\hat{\bm{s}}_2$, but produces tokens with probability below $\theta$ (e.g. uncertain entities or facts). The system intercepts the low-confidence sentence, formulates a query $\bm{q}_2$ anticipating future facts, retrieves external evidence $\mathcal{D}_{\bm{q}_2}$, and regenerates $\bm{s}_2$ conditioned on $[\mathcal{D}_{\bm{q}_2}, \bm{x}, \bm{y}_{<2}]$.
> 4. **Step 3 (Subsequent Regeneration):** The process repeats iteratively until the model completes the entire response.

Retrieval augmented LMs commonly use a retrieve-and-generate setup where they retrieve documents based on the user’s input, and then generate a complete answer conditioning on the retrieved documents Chen et al. (2017); Guu et al. (2020); Lewis et al. (2020); Izacard and Grave (2021); Sachan et al. (2021); Lee et al. (2021); Jiang et al. (2022); Izacard et al. (2022); Nakano et al. (2021); Qian et al. (2023); Lazaridou et al. (2022); Shi et al. (2023).
These single-time retrieval augmented LMs outperform purely parametric LMs, particularly for short-form knowledge-intensive generation tasks such as factoid question answering (QA) Kwiatkowski et al. (2019); Joshi et al. (2017), where *the information needs are clear in the user’s input, and it is sufficient to retrieve relevant knowledge once solely based on the input*.

Increasingly powerful large LMs have also demonstrated abilities in more complex tasks that involve generating long-form output, such as long-form QA Fan et al. (2019); Stelmakh et al. (2022), open-domain summarization Cohen et al. (2021); Hayashi et al. (2021); Giorgi et al. (2022), and (chain-of-thought; CoT) reasoning Wei et al. (2022); Ho et al. (2020); Geva et al. (2021); Hendrycks et al. (2020).
In contrast to short-form generation, long-form generation presents complex information needs that are *not always evident from the input alone*. Similar to how humans gradually gather information as we create content such as papers, essays, or books, long-form generation with LMs would *require gathering multiple pieces of knowledge throughout the generation process*.
For example, to generate a summary about a particular topic, the initial retrieval based on the topic name (e.g., Joe Biden) may not cover all aspects and details.
It is crucial to retrieve extra information as needed during generation, such as when generating a certain aspect (e.g., Joe Biden’s education history) or a specific detail (e.g., the date of Joe Biden’s presidential campaign announcement).

Several attempts have been made to retrieve multiple times throughout generation.
These attempts include methods that passively use the past context to retrieve additional information at a fixed interval Khandelwal et al. (2020); Borgeaud et al. (2022); Ram et al. (2023); Trivedi et al. (2022) which might not accurately reflect what LMs intend to generate in the future or retrieve at inappropriate points.
Some works in multihop QA decompose the full question into sub-questions, each of which is used to retrieve extra information Press et al. (2022); Yao et al. (2022); Khot et al. (2022); Khattab et al. (2022).

We ask the following question: can we create a simple and generic retrieval augmented LM that *actively decides when and what to retrieve* throughout the generation process, and are applicable to a variety of long-form generation tasks?
We provide a generalized view of active retrieval augmented generation.
Our hypothesis regarding *when to retrieve* is that LMs should retrieve information only when they lack the required knowledge to avoid unnecessary or inappropriate retrieval that occurs in passive retrieval augmented LMs Khandelwal et al. (2020); Borgeaud et al. (2022); Ram et al. (2023); Trivedi et al. (2022).
Given the observation that large LMs tend to be well-calibrated and low probability/confidence often indicates a lack of knowledge Kadavath et al. (2022), we adopt an active retrieval strategy that only retrieves when LMs generate low-probability tokens.
When deciding *what to retrieve*, it is important to consider what LMs intend to generate in the future, as the goal of active retrieval is to benefit future generations.
Therefore, we propose anticipating the future by generating a temporary next sentence, using it as a query to retrieve relevant documents, and then regenerating the next sentence conditioning on the retrieved documents.
Combining the two aspects, we propose Forward-Looking Active REtrieval augmented generation (FLARE), as illustrated in Figure 1.
FLARE iteratively generates *a temporary next sentence*, use it as the query to retrieve relevant documents *if it contains low-probability tokens* and regenerate the next sentence until reaches the end.

FLARE is applicable to any existing LMs at inference time without additional training.
Considering the impressive performance achieved by GPT-3.5 Ouyang et al. (2022) on a variety of tasks, we examine the effectiveness of our methods on text-davinci-003.
We evaluate FLARE on 4 diverse tasks/datasets involving generating long outputs, including multihop QA (2WikiMultihopQA), commonsense reasoning (StrategyQA), long-form QA (ASQA), and open-domain summarization (WikiAsp) Ho et al. (2020); Geva et al. (2021); Stelmakh et al. (2022); Hayashi et al. (2021).
Over all tasks, FLARE achieves superior or competitive performance compared to single-time and multi-time retrieval baselines, demonstrating the effectiveness and generalizability of our method.


---

## 2 Retrieval Augmented Generation

We formally define single-time retrieval augmented generation and propose the framework of active retrieval augmented generation.

### 2.1 Notations and Definitions

Given a user input $\bm{x}$ and a document corpus $\mathcal{D}=\{\bm{d}_{i}\}_{i=1}^{|\mathcal{D}|}$ (such as all Wikipedia articles), the goal of retrieval augmented LMs is to generate the answer $\bm{y}=[\bm{s}_{1},\bm{s}_{2},...,\bm{s}_{m}]=[w_{1},w_{2},...,w_{n}]$ containing $m$ sentences or $n$ tokens leveraging information retrieved from the corpus.

In retrieval augmented LM, the LM typically pairs with a retriever that can retrieve a list of documents $\mathcal{D}_{\bm{q}}=\text{ret}(\bm{q})$ for a query $\bm{q}$; the LM conditions on both the user input $\bm{x}$ and retrieved documents $\mathcal{D}_{\bm{q}}$ to generate the answer.
Since we focus on examining various methods of determining when and what to retrieve, we follow existing methods Ram et al. (2023); Trivedi et al. (2022) to prepend the retrieved documents before the user input to aid future generation for both baselines and our method for fair comparisons: $\bm{y}=\text{LM}([\mathcal{D}_{\bm{q}},\bm{x}])$, where $[\cdot,\cdot]$ is concatenation following the specified order.

### 2.2 Single-time Retrieval Augmented Generation

The most common choice is to directly use the user input as the query for retrieval and generate the complete answer at once $\bm{y}=\text{LM}([\mathcal{D}_{\bm{x}},\bm{x}])$.

### 2.3 Active Retrieval Augmented Generation

To aid long-form generation with retrieval, we propose active retrieval augmented generation.
It is a generic framework that actively decides when and what to retrieve through the generation process, resulting in the interleaving of retrieval and generation.
Formally, at step $t(t\geq 1)$, the retrieval query $\bm{q}_{t}$ is formulated based on both the user input $\bm{x}$ and previously generated output $\bm{y}_{<t}=[\bm{y}_{0},...,\bm{y}_{t-1}]$: 
$$
\bm{q}_{t}=\text{qry}(\bm{x},\bm{y}_{<t}),
$$
 where $\text{qry}(\cdot)$ is the query formulation function.
At the beginning ($t=1$), the previous generation is empty ($\bm{y}_{<1}=\emptyset$), and the user input is used as the initial query ($\bm{q}_{1}=\bm{x}$).
Given retrieved documents $\mathcal{D}_{\bm{q}_{t}}$, LMs continually generate the answer until the next retrieval is triggered or reaches the end: 
$$
\bm{y}_{t}=\text{LM}([\mathcal{D}_{\bm{q}_{t}},\bm{x},\bm{y}_{<t}]),
$$
 where $\bm{y}_{t}$ represents the generated tokens at the current step $t$, and the input to LMs is the concatenation of the retrieved documents $\mathcal{D}_{\bm{q}_{t}}$, the user input $\bm{x}$, and the previous generation $\bm{y}_{<t}$.
We discard previously retrieved documents $\cup_{t^{\prime}<t}\mathcal{D}_{\bm{q}_{t^{\prime}}}$ and only use the retrieved documents from the current step to condition the next generation to prevent reaching the input length limit of LMs.


> [!IMPORTANT] Formal Framework: Active Retrieval Augmented Generation
> At each step $t \,(t \ge 1)$:
> 1. **Query Formulation:**
>    $$
>    \bm{q}_{t} = \text{qry}(\bm{x}, \bm{y}_{<t}) \tag{1}
>    $$
>    where at step $t=1$, $\bm{y}_{<1} = \emptyset$ and $\bm{q}_1 = \bm{x}$.
> 2. **Document Retrieval & Conditioning:**
>    $$
>    \bm{y}_{t} = \text{LM}([\mathcal{D}_{\bm{q}_{t}}, \bm{x}, \bm{y}_{<t}]) \tag{2}
>    $$
>    where $\mathcal{D}_{\bm{q}_t} = \text{ret}(\bm{q}_t)$.
> 3. **Memory Management:** Previously retrieved documents $\bigcup_{t' < t} \mathcal{D}_{\bm{q}_{t'}}$ are discarded to prevent exceeding context window limits.


---

## 3 FLARE: Forward-Looking Active REtrieval Augmented Generation

Our intuition is that (1) LMs should only retrieve information when they do not have the necessary knowledge to avoid unnecessary or inappropriate retrieval, and (2) the retrieval queries should reflect the intents of future generations.
We propose two forward-looking active retrieval augmented generation (FLARE) methods to implement the active retrieval augmented generation framework.
The first method prompts the LM to generate retrieval queries when necessary while generating the answer using retrieval-encouraging instructions, denoted as FLARE${}_{\text{instruct}}$.
The second method directly uses the LM’s generation as search queries, denoted as FLARE${}_{\text{direct}}$, which iteratively generates the next sentence to gain insight into the future topic, and if uncertain tokens are present, retrieves relevant documents to regenerate the next sentence.


### Figure 2: Forward-Looking Active Retrieval with Instructions (FLARE_instruct)

> **Figure 2 Caption:** An illustration of forward-looking active retrieval augmented generation with retrieval instructions ($\text{FLARE}_{\text{instruct}}$). It iteratively generates search queries (shown in gray italic) to retrieve relevant information to aid future generations.

> [!NOTE] Figure 2 Structural Breakdown: $\text{FLARE}_{\text{instruct}}$ Mechanism
> - **In-Context Two-Skill Composition:** The LM is prompted with two distinct skills: Skill 1 demonstrates when and how to emit explicit tool calls `[Search(query)]`, and Skill 2 provides exemplars for the target downstream task.
> - **Interleaved Execution:** While generating text, whenever the LM generates a search marker (e.g. `The colors on the flag of Ghana have the following meanings. Red is for [Search(Ghana flag red meaning)]`), decoding halts immediately.
> - **External Search Execution:** The string inside `Search(...)` is parsed and passed to the retriever. The top retrieved documents are inserted into the prompt context, and LM generation resumes conditioning on the newly acquired evidence.

### 3.1 FLARE with Retrieval Instructions

Inspired by Toolformer Schick et al. (2023), a straightforward way of expressing information needs for retrieval is to generate “[Search(query)]” when additional information is needed Schick et al. (2023), e.g., “The colors on the flag of Ghana have the following meanings. Red is for [Search(Ghana flag red meaning)] the blood of martyrs, …”
When working with GPT-3.5 models that offer only API access, we elicit such behavior by few-shot prompting Brown et al. (2020).

Specifically, for a downstream task, we place the search-related instruction and exemplars at the beginning as skill 1, followed by the instruction and exemplars of the downstream task as skill 2.
Given a test case, we ask LMs to combine skills 1 and 2 to generate search queries while performing the task.
The structure of the prompt is shown in Prompt subsection 3.1, and full details can be found in Prompt Appendix D.


> [!TIP] Prompt 3.1: Retrieval Instructions Schema
```text
Prompt 3.1: retrieval instructions


Skill 1. An instruction to guide LMs to generate search queries.
Several search-related exemplars.

Skill 2. An instruction to guide LMs to perform a specific downstream task (e.g., multihop QA).
Several task-related exemplars.

An instruction to guide LMs to combine skills 1 and 2 for the test case.
The input of the test case.
```

As shown in Figure 2, when the LM generates “[Search(query)]” (shown in gray italic), we stop the generation and use the query terms to retrieve relevant documents, which are prepended before the user input to aid future generation until the next search query is generated or reaches the end.
Additional implementation details are included in Appendix A.

### 3.2 Direct FLARE

Since we cannot fine-tune black-box LMs, we found queries generated by FLARE${}_{\text{instruct}}$ through retrieval instructions might not be reliable.
Therefore, we propose a more direct way of forward-looking active retrieval that uses the next sentence to decide when and what to retrieve.

#### 3.2.1 Confidence-based Active Retrieval

As shown in Figure 1, at step $t$, we first generate a temporary next sentence $\hat{\bm{s}}_{t}=\text{LM}([\bm{x},\bm{y}_{<t}])$ without conditioning on retrieved documents.
Then we decide whether to trigger retrieval and formulate queries based on $\hat{\bm{s}}_{t}$.
If the LM is confident about $\hat{\bm{s}}_{t}$, we accept it without retrieving additional information; if not, we use $\hat{\bm{s}}_{t}$ to formulate search queries $\bm{q}_{t}$ to retrieve relevant documents, and then regenerate the next sentence $\bm{s}_{t}$.
The reason we utilize sentences as the basis of our iteration is due to their significance as semantic units that are neither too short nor too lengthy like phrases and paragraphs.
However, our approach can also utilize phrases or paragraphs as the basis.

Since LMs tend to be well-calibrated that low probability/confidence often indicates a lack of knowledge Jiang et al. (2021); Kadavath et al. (2022); Varshney et al. (2022), we actively trigger retrieval if any token of $\hat{\bm{s}}_{t}$ has a probability lower than a threshold $\theta\in[0,1]$.
$\theta=0$ means retrieval is never triggered, while $\theta=1$ triggers retrieval every sentence. 
$$
\bm{y}_{t}=\begin{cases}\hat{\bm{s}}_{t}\quad\quad\text{if all tokens of }\hat{\bm{s}}_{t}\text{ have probs}\geq\theta\\
\bm{s}_{t}=\text{LM}([\mathcal{D}_{\bm{q}_{t}},\bm{x},\bm{y}_{<t}])\quad\quad\text{otherwise}\end{cases}
$$
 where the query $\bm{q}_{t}$ is formulated based on $\hat{\bm{s}}_{t}$.


> [!IMPORTANT] Decision Rule: Confidence-Based Sentence Regeneration
> At step $t$, given temporary sentence $\hat{\bm{s}}_t = \text{LM}([\bm{x}, \bm{y}_{<t}])$:
> $$
> \bm{y}_{t} = \begin{cases}
> \hat{\bm{s}}_{t} & \text{if all tokens of } \hat{\bm{s}}_{t} \text{ have probs} \geq \theta \\
> \bm{s}_{t} = \text{LM}([\mathcal{D}_{\bm{q}_{t}}, \bm{x}, \bm{y}_{<t}]) & \text{otherwise}
> \end{cases} \tag{3}
> $$
> where $\theta \in [0, 1]$ controls retrieval aggressiveness ($\theta=0$: no retrieval; $\theta=1$: retrieve every sentence).

#### 3.2.2 Confidence-based Query Formulation

One way to perform retrieval is to directly use the next sentence $\hat{\bm{s}}_{t}$ as the query $\bm{q}_{t}$.
This shares a similar spirit with methods that use generated hypothetical titles or paragraphs from LMs as retrieval queries or evidences Gao et al. (2022); Sun et al. (2022); Yu et al. (2022); Mao et al. (2021).
We generalize such techniques to long-form generation where active information access is essential.

We found retrieving with the next sentence achieves significantly better results than with the previous context, as shown later in subsection 6.2.
However, it has a risk of perpetuating errors contained in it.
For example, if the LM produces the sentence “Joe Biden attended the University of Pennsylvania” instead of the correct fact that he attended the University of Delaware, using this erroneous sentence as a query might retrieve misleading information.
We propose two simple methods to overcome this issue as illustrated in Figure 3.


### Figure 3: Query Formulation Strategies

> **Figure 3 Caption:** Implicit and explicit query formulation. Tokens with low probabilities are marked with underlines.

> [!NOTE] Figure 3 Detailed Structural Breakdown
> - **Candidate Sentence with Uncertainty:** Suppose the LM generates temporary sentence $\hat{\bm{s}}_t$: *"Joe Biden attended the <u>University of Pennsylvania</u>"*, where "University of Pennsylvania" has low token probabilities ($p < \beta$).
> - **Strategy 1: Masked Sentences (Implicit Query Formulation):**
>   - The uncertain span is masked out: `Joe Biden attended [MASK]`.
>   - Eliminates misleading/hallucinated entities from the query, allowing dense/lexical retrievers to match Joe Biden's true higher education background.
> - **Strategy 2: Generated Questions (Explicit Query Formulation):**
>   - The system extracts uncertain span $\bm{z} = \text{"University of Pennsylvania"}$.
>   - Prompts an auxiliary model (`gpt-3.5-turbo`) to generate a targeted question answering that span: `Which university did Joe Biden attend?`.
>   - Retrieves documents matching the explicit question to provide definitive grounded facts.

##### Masked sentences as implicit queries.

The first method masks out low-confidence tokens in $\hat{\bm{s}}_{t}$ with probabilities below a threshold $\beta\in[0,1]$, where a higher $\beta$ results in more aggressive masking.
This removes potential distractions from the sentence to improve retrieval accuracy.

##### Generated questions as explicit queries.

Another method is to generate explicit questions that target the low-confident span in $\hat{\bm{s}}_{t}$.
For example, if the LM is uncertain about “the University of Pennsylvania”, a question like “Which university did Joe Biden attend?” can help retrieve relevant information.
Self-ask Press et al. (2022) achieved this by manually inserting follow-up questions into downstream task exemplars as shown later in Prompt Appendix D, which requires task-specific annotation efforts.
Instead, we developed a universal approach that generates questions for low-confidence spans without additional annotation.
Specifically, We first extract all spans from $\hat{\bm{s}}_{t}$ with probabilities below $\beta$.
For each extracted span $\bm{z}$, we prompt gpt-3.5-turbo to generate a question $\bm{q}_{t,\bm{z}}$ that can be answered with the span:


> [!TIP] Prompt 3.2: Zero-Shot Question Generation
```text
Prompt 3.2: zero-shot question generation


User input $\bm{x}$.
Generated output so far $\bm{y}_{\leq t}$.

Given the above passage, ask a question to which the answer is the term/entity/phrase “$\bm{z}$”.
```

We retrieve using each generated question and interleave the returned documents into a single ranking list to aid future generations. In summary, queries $\bm{q}_{t}$ are formulated based on $\hat{\bm{s}}_{t}$ as follows:


> [!IMPORTANT] Summary: Direct FLARE Query Formulation
> The search query $\bm{q}_t$ is formulated dynamically based on temporary sentence $\hat{\bm{s}}_t$:
> $$
> \bm{q}_{t} = \begin{cases}
> \emptyset & \text{if all tokens of } \hat{\bm{s}}_{t} \text{ have probs} \geq \theta \\
> \text{mask}(\hat{\bm{s}}_{t}) \text{ or } \text{qgen}(\hat{\bm{s}}_{t}) & \text{otherwise}
> \end{cases} \tag{4}
> $$

### 3.3 Implementation Details

##### Base LM

We validate our method on one of the most advanced GPT-3.5 LMs text-davinci-003 by iteratively querying their API.[^2]

[^2]: `https://api.openai.com/v1/completions` April 23.

##### Document corpus and retrievers.

Since we focus on the integration of retrieval and generation, we use off-the-shelf retrievers that take queries as inputs and return a list of relevant documents. For datasets that mainly rely on knowledge from Wikipedia, we use the Wikipedia dump from Karpukhin et al. (2020) and employ BM25 (Robertson and Zaragoza, 2009) as the retriever. For datasets that rely on knowledge from the open web, we use the Bing search engine as our retriever.[^3]

[^3]: `https://www.microsoft.com/en-us/bing/apis/bing-web-search-api`

##### Retrieved document formatting.

Multiple retrieved documents are linearized according to their ranking and then added to the beginning of the user input using Prompt Appendix D.

Other implementation details such as sentence tokenization and efficiency are included Appendix A.


---

## 4 Multi-time Retrieval Baselines

Existing passive multi-time retrieval augmented LMs can also be formulated using our framework (subsection 2.3).
In this section, we formally introduce three baseline categories based on when and what to retrieve.
These baselines are not exact reproductions of the corresponding paper because many design choices differ which makes direct comparisons impossible.
We implemented them using the same settings, with the only variation being when and what to retrieve.

##### Previous-window

approaches trigger retrieval every $l$ tokens, where $l$ represents the window size. Generated tokens from the previous window are used as the query:


$$
\begin{aligned}
\bm{q}_{t} &= \bm{y}_{t-1} \quad (t \geq 2), \tag{5} \\
\bm{y}_{t} &= [w_{(t-1)l+1}, \dots, w_{tl}]. \tag{6}
\end{aligned}
$$

Some existing methods in this category are RETRO (Borgeaud et al., 2022), IC-RALM (Ram et al., 2023), which retrieve every few tokens, and KNN-LM (Khandelwal et al., 2020), which retrieves every token.[^4] We follow Ram et al. (2023) to use a window size of $l=16$.

[^4]: Since KNN-LM uses the contextualized representation corresponding to the current decoding position to retrieve relevant information which encodes all previous tokens, strictly speaking, $\bm{q}_t$ should be $\bm{y}_{<t}$.

##### Previous-sentence

approaches trigger retrieval every sentence and use the previous sentence as the query, and IRCoT (Trivedi et al., 2022) belongs to this category:


$$
\begin{aligned}
\bm{q}_{t} &= \bm{y}_{t-1} \quad (t \geq 2), \tag{7} \\
\bm{y}_{t} &= \bm{s}_{t}. \tag{8}
\end{aligned}
$$

##### Question decomposition

approaches manually annotated task-specific exemplars to guide LMs to generate decomposed sub-questions while producing outputs.
For example, self-ask Press et al. (2022), a method in this category, manually inserts sub-questions in exemplars using Prompt Appendix D.
For the test case, retrieval is triggered dynamically whenever the model generates a sub-question.

The aforementioned approaches can retrieve additional information while generating.
However, they have notable drawbacks: (1) Using previously generated tokens as queries might not reflect what LMs intend to generate in the future. (2) Retrieving information at a fixed interval can be inefficient because it might occur at inappropriate points. (3) Question decomposition approaches require task-specific prompt engineering, which restricts their generalizability in new tasks.


> [!NOTE] Comparative Analysis: FLARE vs. Multi-time Baselines
> | Dimension | Previous-Window (e.g. RETRO, IC-RALM) | Previous-Sentence (e.g. IRCoT) | Question Decomposition (e.g. Self-Ask) | FLARE (Ours) |
> | :--- | :--- | :--- | :--- | :--- |
> | **When to retrieve** | Fixed token intervals ($l=16$) | Fixed sentence intervals | Whenever decomposed question emitted | **Active / Adaptive** (tokens with prob $< \theta$) |
> | **What to retrieve** | Previous $l$ tokens | Previous sentence | Decomposed sub-question | **Forward-looking** (masked sentence or generated question) |
> | **Task Engineering** | Generic | Generic | Requires custom exemplars per task | **Universal / Generic** |


---

## 5 Experimental Setup

We evaluate the effectiveness of FLARE on 4 diverse knowledge-intensive tasks using few-shot in-context learning Radford et al. (2019); Brown et al. (2020); Liu et al. (2023).
We follow previous works Trivedi et al. (2022) to sub-sample at most 500 examples from each dataset due to the cost of running experiments.
Datasets, metrics, and settings are summarized in Table 7 of Appendix B.
The hyperparameters of FLARE are selected based on the development set and listed in Table 9.
FLARE refers to FLARE${}_{\text{direct}}$ if not specifically stated.

##### Multihop QA

The goal of multihop QA is to answer complex questions through information retrieval and reasoning.
We use 2WikiMultihopQA Ho et al. (2020) which contains 2-hop complex questions sourced from Wikipedia articles that require composition, comparison, or inference, e.g., “Why did the founder of Versus die?”
We follow Wang et al. (2022) to generate both the chain-of-thought and the final answer.
Experimental setting details are included in Appendix B.

We use regular expressions to extract the final answer from the output and compare it with the reference answer using exact match (EM), and token-level F1, precision, and recall.

##### Commonsense reasoning

Commonsense reasoning requires world and commonsense knowledge to generate answers.
We use StrategyQA Geva et al. (2021) which is a collection of crowdsourced yes/no questions, e.g., “Would a pear sink in water?”
We follow Wei et al. (2022) to generate both the chain-of-thought and the final yes/no answer.
Details are included in Appendix B.

We extract the final answer and match it against the gold answer using exact match.

##### Long-form QA

Long-form QA aims to generate comprehensive answers to questions seeking complex information Fan et al. (2019); Stelmakh et al. (2022).
We use ASQA Stelmakh et al. (2022) as our testbed where inputs are ambiguous questions with multiple interpretations, and outputs should cover all of them.
For example, “Where do the Philadelphia Eagles play their home games?” could be asking about the city, sports complex, or stadium.
We found in many cases it is challenging even for humans to identify which aspect of the question is ambiguous.
Therefore, we created another setting (ASQA-hint) where we provide a brief hint to guide LMs to stay on track when generating answers.
The hint for the above case is “This question is ambiguous in terms of which specific location or venue is being referred to.”
Experimental setting details are included in Appendix B.

We use metrics from Stelmakh et al. (2022), including EM, RoBERTa-based QA score (Disambig-F1), ROUGE Lin (2004), and an overall score combining Disambig-F1 and ROUGE (DR).

##### Open-domain summarization

The goal of open-domain summarization is to generate a comprehensive summary about a topic by gathering information from open web Giorgi et al. (2022).
We use WikiAsp Hayashi et al. (2021) which aims to generate aspect-based summaries about entities from 20 domains in Wikipedia, e.g., “Generate a summary about Echo School (Oregon) including the following aspects: academics, history.”
Experimental setting details are included in Appendix B.

Metrics include ROUGE, named entity-based F1, and UniEval Zhong et al. (2022) which measures factual consistency.


---

## 6 Experimental Results

We first report overall results across 4 tasks/datasets and compare the performance of FLARE with all the baselines introduced in section 4.
We then run ablation experiments to study the efficacy of various design choices of our method.

### 6.1 Comparison with Baselines

##### Overall results.

The overall performance of FLARE and baseline across all tasks/datasets are reported in Figure 4.
FLARE outperforms all baseline on all tasks/datasets, indicating that FLARE is a generic method that can effectively retrieve additional information throughout the generation.

Among various tasks, multihop QA shows the most significant improvement.
This is largely due to the task’s clear definition and specific objective of producing the final answer through a 2-hop reasoning process, which makes it easier for LMs to generate on-topic output.
In contrast, ASQA and WikiAsp are more open-ended, which increases the difficulty of both generation and evaluation.
The improvement on ASQA-hint is larger than that of ASQA because identifying ambiguous aspects is challenging even for humans in many cases, and providing a generic hint helps LMs to stay on topic.


### Figure 4: Overall Comparison Across Tasks

```mermaid
xychart-beta
    title "Figure 4: Primary Metric Across Benchmarks"
    x-axis ["2Wiki (EM)", "StrategyQA (EM)", "ASQA (EM)", "ASQA-hint (EM)", "WikiAsp (UniEval)"]
    y-axis "Score (%)" 0 --> 90
    bar [28.2, 72.9, 33.8, 40.1, 47.1]
    bar [39.4, 68.6, 40.0, 43.2, 52.4]
    bar [43.2, 71.2, 39.9, 43.7, 51.8]
    bar [39.0, 71.0, 39.9, 44.7, 52.6]
    bar [51.0, 77.3, 41.3, 46.2, 53.4]
```
*(Bar groups: No retrieval, Single-time retrieval, Previous-window, Previous-sentence, FLARE)*

> **Figure 4 Caption:** Comparison between FLARE and baselines across all tasks/datasets. We report the primary metric for each dataset: EM for 2WikiMultihopQA, StrategyQA, and ASQA, and UniEval for WikiAsp.

> [!NOTE] Figure 4 Detailed Performance Breakdown
> - **2WikiMultihopQA (EM):** No retrieval (28.2) $\to$ Single-time (39.4) $\to$ Previous-window (43.2) $\to$ Question Decomposition (47.8) $\to$ **FLARE (51.0)**. FLARE achieves an absolute gain of **+11.6%** over single-time retrieval and **+3.2%** over heavily engineered question decomposition.
> - **StrategyQA (EM):** No retrieval (72.9) $\to$ Single-time (68.6, noise harms parametric knowledge) $\to$ Previous-window (71.2) $\to$ Previous-sentence (71.0) $\to$ **FLARE (77.3)** (+4.4% over no retrieval, +8.7% over single-time).
> - **ASQA (EM):** No retrieval (33.8) $\to$ Single-time (40.0) $\to$ Previous-window (39.9) $\to$ Previous-sentence (39.9) $\to$ **FLARE (41.3)**.
> - **ASQA-hint (EM):** No retrieval (40.1) $\to$ Single-time (43.2) $\to$ Previous-window (43.7) $\to$ Previous-sentence (44.7) $\to$ **FLARE (46.2)**.
> - **WikiAsp (UniEval):** No retrieval (47.1) $\to$ Single-time (52.4) $\to$ Previous-window (51.8) $\to$ Previous-sentence (52.6) $\to$ **FLARE (53.4)**.

##### Thorough comparisons with baselines.

The performance of all baselines on 2WikiMultihopQA are reported in Table 1.
FLARE outperforms all baselines by a large margin, which confirms that forward-looking active retrieval is highly effective.
Most multi-time retrieval augmented approaches outperform single-time retrieval but with different margins.
The improvement of retrieving using the previous sentence is relatively small which we hypothesize is mainly because the previous sentence often describes entities or relations different from those in the next sentence in 2WikiMultihopQA.
While the previous-window approach might use the first half of a sentence to retrieve information potentially helpful for generating the second half.
Among all baselines, the question decomposition approach Press et al. (2022) achieves the best performance. which is not surprising since the in-context exemplars manually annotated with decomposed sub-questions (Prompt Appendix D) guide LMs to generate sub-questions that align with the topic/intent of future generations.
FLARE outperforms this baseline, indicating that manual exemplar annotation is not necessary for effective future-aware retrieval.
The gap between FLARE${}_{\text{instruct}}$ and question decomposition is large, indicating that teaching LMs to generate search queries using task-generic retrieval instructions and exemplars is challenging.

We report all metrics for the other datasets in Table 2.
FLARE outperforms baselines with respect to all metrics.
Retrieval using the previous window underperforms single-time retrieval on ASQA, which we hypothesize is because the previous window does not accurately reflect future intent.
Since we focus on evaluating factuality, metrics with an emphasis on factual content (such as EM, Disambig-F1, UniEval) are more reliable than metrics computed over all tokens (ROUGE-L).


### Table 1: FLARE and Baselines on 2WikiMultihopQA

| Methods | EM | F1 | Prec. | Rec. |
| :--- | :---: | :---: | :---: | :---: |
| **No retrieval** | 28.2 | 36.8 | 36.5 | 38.6 |
| **Single-time retrieval** | 39.4 | 48.8 | 48.6 | 51.5 |
| *Multi-time retrieval* | | | | |
| Previous-window | 43.2 | 52.3 | 51.7 | 54.5 |
| Previous-sentence | 39.0 | 49.2 | 48.9 | 51.8 |
| Question decomposition | 47.8 | 56.4 | 56.1 | 58.6 |
| $\text{FLARE}_{\text{instruct}}$ (ours) | 42.4 | 49.8 | 49.1 | 52.5 |
| $\text{FLARE}_{\text{direct}}$ (ours) | **51.0** | **59.7** | **59.1** | **62.6** |

> *Table 1 Caption: FLARE and baselines on 2WikiMultihopQA. Previous-window (Borgeaud et al., 2022; Ram et al., 2023), previous-sentence (Trivedi et al., 2022), and question decomposition (Press et al., 2022; Yao et al., 2022) methods are reimplemented for fair comparisons.*


### Table 2: Comparison Between FLARE and Baselines Across Tasks

| Datasets | StrategyQA | ASQA | | | | ASQA-hint | | | | WikiAsp | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Metrics** | **EM** | **EM** | **D-F1** | **R-L** | **DR** | **EM** | **D-F1** | **R-L** | **DR** | **UniEval** | **E-F1** | **R-L** |
| No retrieval | 72.9 | 33.8 | 24.2 | 33.3 | 28.4 | 40.1 | 32.5 | 36.4 | 34.4 | 47.1 | 14.1 | 26.4 |
| Single-time retrieval | 68.6 | 40.0 | 27.1 | 34.0 | 30.4 | 43.2 | 34.8 | 37.4 | 36.0 | 52.4 | 17.4 | 26.9 |
| *Multi-time retrieval* | | | | | | | | | | | | |
| Previous-window | 71.2 | 39.9 | 27.0 | 34.3 | 30.4 | 43.7 | 35.7 | 37.5 | 36.6 | 51.8 | 18.1 | 27.3 |
| Previous-sentence | 71.0 | 39.9 | 27.9 | 34.3 | 30.9 | 44.7 | 35.9 | 37.5 | 36.7 | 52.6 | 17.8 | 27.2 |
| **FLARE (ours)** | **77.3** | **41.3** | **28.2** | **34.3** | **31.1** | **46.2** | **36.7** | **37.7** | **37.2** | **53.4** | **18.9** | **27.6** |

> *Table 2 Caption: Comparison between FLARE and baselines on StrategyQA, ASQA, ASQA-hint, and WikiAsp. D-F1 is Disambig-F1, R-L is ROUGE-L, and E-F1 is named entity-based F1.*

### 6.2 Ablation Study

##### Importance of forward-looking retrieval.

We first validate that forward-looking retrieval is more effective than past-context-based retrieval.
We run ablation experiments on 2WikiMultihopQA and ASQA-hint comparing retrieval using the previous versus the next sentence.
Specifically, both methods retrieve every sentence and directly use the complete previous/next sentence as queries.
As shown in Table 3, using the next sentence to retrieve is clearly better than using the previous sentence, confirming our hypothesis.


### Table 3: Head-to-Head Comparison: Previous Sentence vs. Next Sentence Retrieval

| Retrieval Query | 2WikiMultihopQA | | | | ASQA-hint | | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **EM** | **F1** | **Prec.** | **Rec.** | **EM** | **D-F1** | **R-L** | **DR** |
| Previous sentence | 39.0 | 49.2 | 48.9 | 51.8 | 42.5 | 34.1 | 36.9 | 35.5 |
| **Next sentence (FLARE)** | **48.8** | **57.6** | **57.1** | **60.5** | **45.9** | **35.7** | **37.5** | **36.6** |

> *Table 3 Caption: A head-to-head comparison between using the previous sentence and the next sentence for retrieval.*

We also run previous-window approaches using different numbers of past tokens as queries.
As shown in Table 4, using too many tokens ($>32$) in the past hurts the performance, further confirming our hypothesis that previous context might not be relevant to intent of future generations.


### Table 4: Previous-Window Approaches with Different Token Context Lengths

| #Tokens | EM | F1 | Prec. | Rec. |
| :---: | :---: | :---: | :---: | :---: |
| 16 | 43.2 | 52.3 | 51.7 | 54.5 |
| **32** | **43.6** | **52.4** | **52.0** | **55.0** |
| 48 | 40.0 | 49.3 | 49.0 | 52.0 |
| All | 39.0 | 48.5 | 48.2 | 51.1 |

> *Table 4 Caption: Previous-window approaches using different numbers of tokens as queries.*

##### Importance of active retrieval.

Next, we investigate how active retrieval threshold $\theta$ affects performance.
To alter our method from not retrieving to retrieving every sentence, we adjust the confidence threshold $\theta$ that determines when to trigger retrieval from 0 to 1.
We then calculate the proportion of steps/sentences where retrieval is activated, and present the performance based on it.
As shown in Figure 5, on 2WikiMultihopQA, the performance plateaus when the retrieval percentage exceeds 60%, indicating that retrieval when LMs are confident is not necessary.
On StrategyQA, the performance drops when the retrieval percentage exceeds 50%, indicating that unnecessary retrieval can introduce noise and impede the original generation process.
We found triggering retrieval for 40%-80% of sentences usually leads to a good performance across tasks/datasets.


### Figure 5: Sensitivity to Retrieval Percentage (Threshold $\theta$)

> **Figure 5 Caption:** Performance (EM) of FLARE with respect to the percentage of steps/sentences with retrieval on 2WikiMultihopQA and StrategyQA.

> [!NOTE] Figure 5 Detailed Analysis
> - **2WikiMultihopQA (Complex Multihop Reasoning):** EM rises sharply from 28.2% at 0% retrieval up to ~51% as retrieval percentage increases towards 60%, after which the curve plateaus. Confident steps do not need additional evidence.
> - **StrategyQA (Commonsense Reasoning):** EM increases from 72.9% at 0% retrieval to a peak of 77.3% at ~40–50% retrieval. Beyond 50%, over-retrieval causes performance to plunge below 70%, showing that injecting unneeded retrieved documents introduces distractor noise that degrades parametric reasoning.
> - **Practical Recommendation:** Triggering retrieval for **40%–80%** of sentences yields robust, optimal performance across tasks.

##### Effectiveness of different query formulation methods

We study implicit query formation by masking and explicit query formulation through question generation.
In Table 5, we compare the performance of FLARE with different masking thresholds $\beta$.
Retrieving directly with the complete sentence ($\beta=0$) is worse than masking tokens with low probabilities, confirming our hypothesis that low-confidence erroneous tokens can distract retrievers.
We compare implicit and explicit query formulation methods in Table 6.
Performances of both methods are similar, indicating that both methods can effectively reflect information needs.


### Table 5: Effect of Masking Threshold $\beta$ on 2WikiMultihopQA

| $\beta$ | EM | F1 | Prec. | Rec. |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 (No masking) | 0.488 | 0.576 | 0.571 | 0.605 |
| 0.2 | 0.498 | 0.588 | 0.582 | 0.616 |
| **0.4** | **0.510** | **0.597** | **0.591** | **0.627** |
| 0.6 | 0.506 | 0.593 | 0.586 | 0.622 |

> *Table 5 Caption: Performance of FLARE with respect to the masking threshold $\beta$ on 2WikiMultihopQA.*


### Table 6: Comparison Between Implicit and Explicit Query Formulation

| Query Formulation | ASQA-hint | | | | WikiAsp | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **EM** | **D-F1** | **R-L** | **DR** | **UniEval** | **E-F1** | **R-L** |
| **Implicit (Masking)** | 45.7 | **36.9** | 37.7 | **37.3** | **53.4** | 18.8 | **27.7** |
| **Explicit (Question Gen)** | **46.2** | 36.7 | 37.7 | 37.2 | **53.4** | **18.9** | 27.6 |

> *Table 6 Caption: A comparison between implicit and explicit query formulation methods in FLARE.*


---

## 7 Related Work

We refer to subsection 2.2 and section 4 for extensively discussion on single-time and multi-time retrieval augmented LMs, which is the most relevant area to this paper.

##### Iterative and adaptive retrieval

Iterative retrieval and refinement has been studied in both text and code generation tasks Peng et al. (2023); Zhang et al. (2023); Zemlyanskiy et al. (2022); Yu et al. (2023).
FLARE differs from these methods in the granularity of generation and retrieval strategies.
Adaptive retrieval has been studied in single-time retrieval scenarios based on either question popularity or generation probabilities Mallen et al. (2022); Li et al. (2023), while we focus on long-form generation requiring active information access.

##### Browser-enhanced LMs

WebGPT Nakano et al. (2021) and WebCPM Qin et al. (2023) train LMs to interact with browser to enhance factuality using reinforcement learning or supervised training where multiple queries can be triggered before generation.
FLARE is built on text-based retrievers but can be combined with a browser to potentially improve retrieval quality.


---

## 8 Conclusion

To aid long-form generation with retrieval augmentation, we propose an active retrieval augmented generation framework that decides when and what to retrieve during generation.
We implement this framework with forward-looking active retrieval that iteratively uses the upcoming sentence to retrieve relevant information if it contains low-confidence tokens and regenerates the next sentence.
Experimental results on 4 tasks/datasets demonstrate the effectiveness of our methods.
Future directions include better strategies for active retrieval and developing efficient LM architectures for active information integration.


---

## 9 Limitations

We also conduct experiments on Wizard of Wikipedia Dinan et al. (2019) and ELI5 Fan et al. (2019), and found that FLARE did not provide significant gains.
Wizard of Wikipedia is a knowledge-intensive dialogue generation dataset where the output is relatively short ($\sim$20 tokens on average) so retrieving multiple disparate pieces of information might not be necessary.
ELI5 Fan et al. (2019) is a long-form QA dataset requiring in-depth answers to open-ended questions.
Due to issues mentioned in Krishna et al. (2021) such as difficulties of grounding generation in retrieval and evaluation, both single-time retrieval and FLARE did not provide significant gains over not using retrieval.
From an engineering perspective, interleaving generation and retrieval with a naive implementation increases both overheads and the cost of generation.
LMs need to be activated multiple times (once for each retrieval) and a caching-free implementation also requires recomputing the previous activation each time after retrieval.
This issue can be potentially alleviated with special architectural designs that encode the retrieved documents $\mathcal{D}_{\bm{q}_{t}}$ and the input/generation ($\bm{x}$/$\bm{y}_{<t}$) independently.


---

## Acknowledgements

This work was supported in part by a grant from the Singapore Defence Science and Technology Agency and the IBM PhD Fellowship.
We thank Chunting Zhou, Amanda Bertsch, Uri Alon, Hiroaki Hayashi, Harsh Trivedi, Patrick Lewis, Timo Schick, Kaixin Ma, Shuyan Zhou, and Songwei Ge for their insightful discussions and help with the experiments.


---

## Appendix A FLARE Implementation Details

##### $\text{FLARE}_{\text{instruct}}$ implementation details

We found that LMs can effectively combine retrieval and downstream task-related skills and generate meaningful search queries while performing the task.
However, there are two issues: (1) LMs tend to generate fewer search queries than necessary. (2) Generating excessive search queries can disrupt answer generation and adversely affect performance.
We address these issues using two methods respectively.
First, we increase the logit of the token “[” by 2.0 to improve the chances of LMs generating “[Search(query)]”.
Second, whenever LMs generate a search query, we use it to retrieve relevant information, promptly remove it from the generation, and generate the next few tokens while forbidding “[” by adding a large negative value to the logit of “[”.

##### The initial query of FLARE.

FLARE starts with the user input $\bm{x}$ as the initial query to retrieve documents to generate the first sentence $\hat{\bm{s}}_{1}=\text{LM}([\mathcal{D}_{\bm{x}},\bm{x}])$ to bootstrap the iterative generation process.
For the following steps, the temporary forward-looking sentence is generated without retrieved documents.

##### Sentence tokenization.

For each step $t$, we generate 64 tokens which are longer than most sentences, and use the [NLTK PunktSentenceTokenizer](https://www.nltk.org/api/nltk.tokenize.PunktSentenceTokenizer.html) to extract the first sentence and discard the rest.

##### Efficiency

As shown in subsection 6.2, on average retrieval is triggered for $30\%\sim 60\%$ of sentences depending on downstream tasks.
In comparision, KNN-LM Khandelwal et al. (2020) retrieves every token, RETRO or IC-RALM Borgeaud et al. (2022); Ram et al. (2023) retrievers every 4$\sim$32 tokens, and IRCoT Trivedi et al. (2022) retrieves every sentence.
Compared to single-time retrieval, however, interleaving retrieval and generation with a naive implementation indeed increases overheads, which we discuss in the limitation section (section 9).


---

## Appendix B Datasets and Settings

Datasets, metrics, and experimental settings are summarized in Table 7.


### Table 7: Dataset Statistics and Experimental Settings

| Settings | 2WikiMultihopQA | StrategyQA | ASQA | WikiAsp |
| :--- | :--- | :--- | :--- | :--- |
| **Source Reference** | Ho et al. (2020) | Geva et al. (2021) | Stelmakh et al. (2022) | Hayashi et al. (2021) |
| **Dataset statistics** | | | | |
| Task | multihop QA | commonsense QA | long-form QA | open-domain summarization |
| #Examples | 500 | 229 | 500 | 500 |
| **Evaluation settings** | | | | |
| Metrics | EM, F1, Prec., Rec. | EM | EM, Disambig-F1, ROUGE, DR | UniEval, entity-F1, ROUGE |
| **Retrieval settings** | | | | |
| Corpus | Wikipedia | Wikipedia | Wikipedia | open web |
| Retriever | BM25 | BM25 | BM25 | Bing |
| Top-$k$ | 2 | 3 | 3 | 5 |
| **Prompt format** | | | | |
| #Exemplars | 8 | 6 | 8 | 4 |
| Ret. for exemplars | ✓ | ✗ | ✗ | ✗ |

> *Table 7 Caption: Dataset statistics and experimental settings of different tasks.*

##### Multihop QA

For “Why did the founder of Versus die?”, the output we aim to generate is “The founder of Versus was Gianni Versace. Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997. So the answer is shot.”
We use 8 exemplars from Trivedi et al. (2022) listed in Prompt Appendix D for in-context learning, BM25 as the retriever, and Wikipedia articles as the retrieval corpus.
Similar to the observation in Trivedi et al. (2022), we found incorporating retrieval results for exemplars improves the performance, we use the input $\bm{x}$ of each exemplar to retrieve several documents and then add them using the format in Prompt Appendix D.
We found increasing the number of retrieval documents often increases performance.
Therefore, we use the maximum number of documents that can fit within the input length limit of text-davinci-003, which is 2 for 2WikiMultihopQA.

##### Commonsense Reasoning

For “Would a pear sink in water?”, the output we aim to generate is “The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the final answer is no.”
We use 6 exemplars from Wei et al. (2022) listed in Prompt Appendix D, BM25 on the Wikipedia corpus, and 3 retrieved documents to run experiments.

##### Long-form QA

For “Where do the Philadelphia Eagles play their home games?”, the output we aim to generate is “We need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations and the answers are: (1) The city is Philadelphia. (2) The sports complex is the South Philadelphia Sports Complex. (3) The stadium is the Lincoln Financial Field stadium.”
For both the original setting (ASQA) and the setting with hints (ASQA-hint), we manually annotate 8 exemplars (Prompt Appendix D and Appendix D), use BM25 on the Wikipedia corpus, and 3 retrieved documents to run experiments.

##### Open-domain Summarization

The original WikiAsp dataset is designed for multi-document summarization and provides a list of references to systems.
We converted it into the open-domain setting by removing the associated references and instead gathering information from the open web.
For “Generate a summary about Echo School (Oregon) including the following aspects: academics, history.”, the output we aim to generate is “# Academics. In 2008, 91% of the school’s seniors received their high school diploma… # History. The class of 2008 was the 100th class in the school’s history.” where # is used to indicate aspects.
We manually annotate 4 exemplars (Prompt Appendix D), and use the Bing search engine to retrieve 5 documents from the open web.
To avoid leaking, we exclude several Wikipedia-related domains listed in Table 8 from Bing’s search results.


### Table 8: Wikipedia-Related Domains Excluded from Bing Search Results

```text
wikipedia.org, wikiwand.com, wiki2.org, wikimedia.org
```

> *Table 8 Caption: Wikipedia-related domains excluded from Bing’s search results.*


---

## Appendix C Hyperparameters

Hyperparameters of FLARE on different datasets are listed in Table 9.


### Table 9: Hyperparameters of FLARE on Different Datasets

| Dataset | $\theta$ | $\beta$ | Query formulation | Combine single- & multi-time retrieval |
| :--- | :---: | :---: | :---: | :---: |
| 2WikiMultihopQA | 0.8 | 0.4 | implicit | ✗ |
| StrategyQA | 0.4 | 0.4 | implicit | ✗ |
| ASQA & ASQA-hint | 0.8 | 0.4 | explicit | ✓ |
| WikiAsp | 0.8 | 0.4 | explicit | ✓ |

> *Table 9 Caption: Hyperparameters of FLARE on different datasets.*


---

## Appendix D Prompts and Few-shot Exemplars

The prompt used to linearize multiple documents is shown in Prompt Appendix D.
The prompt used in self-ask Press et al. (2022) is shown in Prompt Appendix D.
Prompts and exemplars of different tasks/datasets are shown in Prompt Appendix D, Appendix D, Appendix D, Appendix D, Appendix D, and Appendix D, respectively.

### Prompt D.1: Document Formatting

```text
Prompt D.1: document formatting


Search results:
$[$1$]$ Document 1
$[$2$]$ Document 2
…
The user input $\bm{x}$
```

### Prompt D.2: Multihop QA with Self-Ask

```text
Prompt D.2: multihop QA with self-ask


Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
Are follow up questions needed here: Yes.
Follow up: How old was Theodor Haecker when he died?
Intermediate answer: Theodor Haecker was 65 years old when he died.
Follow up: How old was Harry Vaughan Watkins when he died?
Intermediate answer: Harry Vaughan Watkins was 69 years old when he died.
So the final answer is: Harry Vaughan Watkins.
```

### Prompt D.3: Retrieval Instructions for 2WikiMultihopQA

```text
Prompt D.3: retrieval instructions for 2WikiMultihopQA


Skill 1. Use the Search API to look up relevant information by writing “[Search(term)]” where “term” is the search term you want to look up. For example:

Question: But what are the risks during production of nanomaterials?
Answer (with Search): [Search(nanomaterial production risks)] Some nanomaterials may give rise to various kinds of lung damage.

Question: The colors on the flag of Ghana have the following meanings.
Answer (with Search): Red is for [Search(Ghana flag red meaning)] the blood of martyrs, green for forests, and gold for mineral wealth.

Question: Metformin is the first-line drug for what?
Answer (with Search): [Search(Metformin first-line drug)] patients with type 2 diabetes and obesity.

Skill 2. Answer questions by thinking step-by-step. First, write out the reasoning steps, then draw the conclusion. For example:

Question: When did the director of film Hypocrite (Film) die?
Answer (with step-by-step): The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.

Question: Are both Kurram Garhi and Trojkrsti located in the same country?
Answer (with step-by-step): Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.

Question: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
Answer (with step-by-step): Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan’s nationality is India. Karl Freund’s nationality is Germany. Thus, they do not have the same nationality. So the answer is no.

Question: Who is Boraqchin (Wife Of Ögedei)’s father-in-law?
Answer (with step-by-step): Boraqchin is married to Ögedei Khan. Ögedei Khan’s father is Genghis Khan. Thus, Boraqchin’s father-in-law is Genghis Khan. So the answer is Genghis Khan.

Question: Who was born first out of Martin Hodge and Ivania Martinich?
Answer (with step-by-step): Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is Martin Hodge.

Question: When did the director of film Laughter In Hell die?
Answer (with step-by-step): The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is August 25, 1963.

Question: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?
Answer (with step-by-step): The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two. So the answer is Twenty Plus Two.

Question: Who is the grandchild of Krishna Shah (Nepalese Royal)?
Answer (with step-by-step): Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah. So the answer is Prithvipati Shah.

Now, combine the aforementioned two skills. First, write out the reasoning steps, then draw the conclusion, where the reasoning steps should also utilize the Search API “[Search(term)]” whenever possible.

Question: Where did Minbyauk Thihapate’s wife die?
Answer (with step-by-step & Search):
```

### Prompt D.4: Exemplars of 2WikiMultihopQA

```text
Prompt D.4: exemplars of 2WikiMultihopQA


Question: When did the director of film Hypocrite (Film) die?
Answer: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.

Question: Are both Kurram Garhi and Trojkrsti located in the same country?
Answer: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.

Question: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
Answer: Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan’s nationality is India. Karl Freund’s nationality is Germany. Thus, they do not have the same nationality. So the answer is no.

Question: Who is Boraqchin (Wife Of Ögedei)’s father-in-law?
Answer: Boraqchin is married to Ögedei Khan. Ögedei Khan’s father is Genghis Khan. Thus, Boraqchin’s father-in-law is Genghis Khan. So the answer is Genghis Khan.

Question: Who was born first out of Martin Hodge and Ivania Martinich?
Answer: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is Martin Hodge.

Question: When did the director of film Laughter In Hell die?
Answer: The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is August 25, 1963.

Question: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?
Answer: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two. So the answer is Twenty Plus Two.

Question: Who is the grandchild of Krishna Shah (Nepalese Royal)?
Answer: Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah. So the answer is Prithvipati Shah.

Question: Which country the director of film Citizen Mavzik is from?
Answer:
```

### Prompt D.5: Exemplars of StrategyQA

```text
Prompt D.5: exemplars of StrategyQA


Generate a yes or no answer to the following question.
Question: Do hamsters provide food for any animals?
Answer: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the final answer is yes.

Generate a yes or no answer to the following question.
Question: Could Brooke Shields succeed at University of Pennsylvania?
Answer: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the final answer is yes.

Generate a yes or no answer to the following question.
Question: Hydrogen’s atomic number squared exceeds number of Spice Girls?
Answer: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the final answer is no.

Generate a yes or no answer to the following question.
Question: Is it common to see frost during some college commencements?
Answer: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the final answer is yes.

Generate a yes or no answer to the following question.
Question: Could a llama birth twice during War in Vietnam (1945-46)?
Answer: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the final answer is no.

Generate a yes or no answer to the following question.
Question: Would a pear sink in water?
Answer: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the final answer is no.

Generate a yes or no answer to the following question.
Question: Did Spiderman fight against Falcon in the MCU?
Answer:
```

### Prompt D.6: Exemplars of ASQA

```text
Prompt D.6: exemplars of ASQA


Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: Who played bonnie in gone with the wind?
Answer: This question is ambiguous in terms of which version or adaptation of Gone with the Wind is being referred to. In order to figure out its interpretations, we need to consider different versions or adaptations of Gone with the Wind. Gone with the Wind has two versions or adaptations: the 1939 film Gone with the Wind or the 2008 musical Gone with the Wind. Therefore, this question has 2 interpretations: (1) Who played Bonnie in the 1939 film Gone with the Wind? (2) Who played Bonnie in the 2008 musical Gone with the Wind? The answers to all interpretations are: (1) The 1939 film Gone with the Wind’s character Bonnie was played by Eleanore Cammack "Cammie" King. (2) The 2008 musical Gone with the Wind’s character Bonnie was played by Leilah de Meza.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: What is the second largest city in the usa?
Answer: This question is ambiguous in terms of the criteria being used to determine the second largest city in the USA. In order to figure out its interpretations, we need to consider different criteria to determine a city’s size. City size can be measured by two criteria: population or area. Therefore, this question has 2 interpretations: (1) What is the second largest city in the USA by population? (2) What is the second largest city in the USA by area? The answers to all interpretations are: (1) The second largest city in the USA by population is Los Angeles, California. (2) The second largest city in the USA by area is Juneau, Alaska.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: When was bohemian rhapsody released as a single?
Answer: This question is ambiguous in terms of which country’s release of the single is being referred to. In order to figure out its interpretations, we need to consider different countries where Bohemian Rhapsody is released. Bohemian Rhapsody was released in the United Kingdom and in the United States on different dates. Therefore, this question has 2 interpretations: (1) When was Bohemian Rhapsody released as a single in the United Kingdom? (2) When was Bohemian Rhapsody released as a single in the United States? The answers to all interpretations are: (1) Bohemian Rhapsody was released as a single in the United Kingdom on 31 October 1975. (2) Bohemian Rhapsody was released as a single in the United States on December 1975.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: Where do the philadelphia eagles play their home games?
Answer: This question is ambiguous in terms of which specific location or venue is being referred to. In order to figure out its interpretations, we need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations: (1) What city do the Philadelphia Eagles play their home games? (2) In what sports complex do the Philadelphia Eagles play their home games? (3) What stadium do the Philadelphia Eagles play their home games? The answers to all interpretations are: (1) Philadelphia Eagles play their home games in the city Philadelphia. (2) Philadelphia Eagles play their home games in the South Philadelphia Sports Complex. (3) Philadelphia Eagles play their home games in the Lincoln Financial Field stadium.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: When did xbox one come out in australia?
Answer: This question is ambiguous in terms of which specific version of the Xbox One is being referred to. In order to figure out its interpretations, we need to consider the different versions of the Xbox One that have been released. Xbox One has two versions: the Xbox One video game console or the Xbox One X high-end model. Therefore, this question has 2 interpretations: (1) When did the Xbox One release in Australia? (2) When did the Xbox One X release in Australia? The answers to all interpretations are: (1) The Xbox One video game console was released in Australia on November 22, 2013. (2) The Xbox One X video game console was released in Australia on November 7, 2017.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: When does the movie summer of 84 come out?
Answer: This question is ambiguous in terms of which release of the movie is being referred to. In order to figure out its interpretations, we need to consider different releases of the movie Summer of ’84. The movie Summer of ’84 is first released at the Sundance Festival before it’s released throughout the US. Therefore, this question has 2 interpretations: (1) When did the movie Summer of ’84 first release at the Sundance Festival? (2) When did the movie Summer of ’84 first release throughout the US? The answers to all interpretations are: (1) Summer of ’84 was released at the Sundance Festival on January 22, 2018. (2) Summer of ’84 was released throughout the US on August 10, 2018.
```

### Prompt D.7: Exemplars of ASQA (cont.)

```text
Prompt D.7: exemplars of ASQA (cont.)


Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: What was roy orbison’s first number one hit?
Answer: This question is ambiguous in terms of which specific chart or region is being referred to. In order to figure out its interpretations, we need to consider the different charts and regions where Roy Orbison’s music was popular. Roy Orbison is popular in both the US Hot 100 and Canada, and the UK and Ireland. Therefore, this question has 2 interpretations: (1) What was Roy Orbison’s first number one hit in the US Hot 100 and Canada? (2) What was Roy Orbison’s first number one hit in the UK and Ireland? The answers to all interpretations are: (1) Running Scared was the first number one hit for Roy Orbison in the US Hot 100 and Canada. (2) Only the Lonely (Know the Way I Feel) was the first number one hit for Roy Orbison in the UK and Ireland.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: What is the criminal’s name in the breakfast club?
Answer: This question is ambiguous in terms of which specific name is being referred to - the character’s name or the actor’s name. In order to figure out its interpretations, we need to consider both possibilities: the character’s name or the actor’s name. Therefore, this question has 2 interpretations: (1) What is the criminal’s character name in The Breakfast Club? (2) What is the the name of the actor who played the criminal in The Breakfast Club? The answers to all interpretations are: (1) John Bender was the name of the criminal’s character in The Breakfast Club. (2) Judd Nelson was the actor of the criminal in The Breakfast Club.

Given an ambiguous question, figure out its interpretations and answer them one by one.
Question: How many state parks are there in virginia?
Answer:
```

### Prompt D.8: Exemplars of ASQA-hint

```text
Prompt D.8: exemplars of ASQA-hint


Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: Who played bonnie in gone with the wind?
Hint: This question is ambiguous in terms of which version or adaptation of Gone with the Wind is being referred to.
Answer: In order to figure out its interpretations, we need to consider different versions or adaptations of Gone with the Wind. Gone with the Wind has two versions or adaptations: the 1939 film Gone with the Wind or the 2008 musical Gone with the Wind. Therefore, this question has 2 interpretations: (1) Who played Bonnie in the 1939 film Gone with the Wind? (2) Who played Bonnie in the 2008 musical Gone with the Wind? The answers to all interpretations are: (1) The 1939 film Gone with the Wind’s character Bonnie was played by Eleanore Cammack "Cammie" King. (2) The 2008 musical Gone with the Wind’s character Bonnie was played by Leilah de Meza.

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: What is the second largest city in the usa?
Hint: This question is ambiguous in terms of the criteria being used to determine the second largest city in the USA.
Answer: In order to figure out its interpretations, we need to consider different criteria to determine a city’s size. City size can be measured by two criteria: population or area. Therefore, this question has 2 interpretations: (1) What is the second largest city in the USA by population? (2) What is the second largest city in the USA by area? The answers to all interpretations are: (1) The second largest city in the USA by population is Los Angeles, California. (2) The second largest city in the USA by area is Juneau, Alaska.

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: When was bohemian rhapsody released as a single?
Hint: This question is ambiguous in terms of which country’s release of the single is being referred to.
Answer: In order to figure out its interpretations, we need to consider different countries where Bohemian Rhapsody is released. Bohemian Rhapsody was released in the United Kingdom and in the United States on different dates. Therefore, this question has 2 interpretations: (1) When was Bohemian Rhapsody released as a single in the United Kingdom? (2) When was Bohemian Rhapsody released as a single in the United States? The answers to all interpretations are: (1) Bohemian Rhapsody was released as a single in the United Kingdom on 31 October 1975. (2) Bohemian Rhapsody was released as a single in the United States on December 1975.

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: Where do the philadelphia eagles play their home games?
Hint: This question is ambiguous in terms of which specific location or venue is being referred to.
Answer: In order to figure out its interpretations, we need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations: (1) What city do the Philadelphia Eagles play their home games? (2) In what sports complex do the Philadelphia Eagles play their home games? (3) What stadium do the Philadelphia Eagles play their home games? The answers to all interpretations are: (1) Philadelphia Eagles play their home games in the city Philadelphia. (2) Philadelphia Eagles play their home games in the South Philadelphia Sports Complex. (3) Philadelphia Eagles play their home games in the Lincoln Financial Field stadium.

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: When did xbox one come out in australia?
Hint: This question is ambiguous in terms of which specific version of the Xbox One is being referred to.
Answer: In order to figure out its interpretations, we need to consider the different versions of the Xbox One that have been released. Xbox One has two versions: the Xbox One video game console or the Xbox One X high-end model. Therefore, this question has 2 interpretations: (1) When did the Xbox One release in Australia? (2) When did the Xbox One X release in Australia? The answers to all interpretations are: (1) The Xbox One video game console was released in Australia on November 22, 2013. (2) The Xbox One X video game console was released in Australia on November 7, 2017.

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: When does the movie summer of 84 come out?
Hint: This question is ambiguous in terms of which release of the movie is being referred to.
Answer: In order to figure out its interpretations, we need to consider different releases of the movie Summer of ’84. The movie Summer of ’84 is first released at the Sundance Festival before it’s released throughout the US. Therefore, this question has 2 interpretations: (1) When did the movie Summer of ’84 first release at the Sundance Festival? (2) When did the movie Summer of ’84 first release throughout the US? The answers to all interpretations are: (1) Summer of ’84 was released at the Sundance Festival on January 22, 2018. (2) Summer of ’84 was released throughout the US on August 10, 2018.
```

### Prompt D.9: Exemplars of ASQA-hint (cont.)

```text
Prompt D.9: exemplars of ASQA-hint (cont.)


Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: What was roy orbison’s first number one hit?
Hint: This question is ambiguous in terms of which specific chart or region is being referred to.
Answer: In order to figure out its interpretations, we need to consider the different charts and regions where Roy Orbison’s music was popular. Roy Orbison is popular in both the US Hot 100 and Canada, and the UK and Ireland. Therefore, this question has 2 interpretations: (1) What was Roy Orbison’s first number one hit in the US Hot 100 and Canada? (2) What was Roy Orbison’s first number one hit in the UK and Ireland? The answers to all interpretations are: (1) Running Scared was the first number one hit for Roy Orbison in the US Hot 100 and Canada. (2) Only the Lonely (Know the Way I Feel) was the first number one hit for Roy Orbison in the UK and Ireland.

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: What is the criminal’s name in the breakfast club?
Hint: This question is ambiguous in terms of which specific name is being referred to - the character’s name or the actor’s name.
Answer: In order to figure out its interpretations, we need to consider both possibilities: the character’s name or the actor’s name. Therefore, this question has 2 interpretations: (1) What is the criminal’s character name in The Breakfast Club? (2) What is the the name of the actor who played the criminal in The Breakfast Club? The answers to all interpretations are: (1) John Bender was the name of the criminal’s character in The Breakfast Club. (2) Judd Nelson was the actor of the criminal in The Breakfast Club.

Given an ambiguous question and a hint on which aspect of the question is ambiguous, figure out its interpretations and answer them one by one.
Question: How many state parks are there in virginia?
Hint: This question is ambiguous in terms of the time frame or period being referred to.
Answer:
```

### Prompt D.10: Exemplars of WikiAsp

```text
Prompt D.10: exemplars of WikiAsp


Generate a summary about Aslanhane Mosque including the following aspects: location, history with one aspect per line.
# Location
The mosque is in the old quarter of ankara next to ankara castle. With an altitude of 947 metres (3,107 ft) it overlooks ankara at 39°56’12"N 32°51’55"E.
# History
The mosque is one of the oldest mosques in Turkey still standing. It was built during the reign of Mesud II of the Anatolian Seljuks in 1290. Its architect was Ebubekir Mehmet. It was commissioned by two Ahi leaders named Hüsamettin and Hasaneddin. However, in 1330, it was repaired by another Ahi leader named Şerafettin after whom the mosque was named. After several minor repairs the mosque was restored by the directorate general of foundations in 2010-2013 term.

Generate a summary about Untold Legends: The Warrior’s Code including the following aspects: reception, gameplay, development with one aspect per line.
# Reception
The game received "mixed or average reviews" according to video game review aggregator Metacritic.
# Gameplay
The warrior’s code is a hack n’ slash action role-playing game, which concentrates on action-oriented combat.
# Development
As a pre-order bonus, the game was shipped with a small action figure of the Guardian class.

Generate a summary about Raid on St. Augustine including the following aspects: aftermath, background with one aspect per line.
# Aftermath
Once the English had gone Menéndez and the rest of the Spanish settlers returned to find a smoldering ruins and very little left. He soon and begged for help from the viceroy of Cuba and the settlement took a while to build itself back up. The destroyed fort was replaced with the present day Castillo de San Marcos.
# Background
War had already been unofficially declared by Philip II of Spain after the Treaty of Nonsuch in which Elizabeth I had offered her support to the rebellious Protestant Dutch rebels. The Queen through Francis Walsingham ordered Sir Francis Drake to lead an expedition to attack the Spanish New World in a kind of preemptive strike. Sailing from Plymouth, England, he struck first at Santiago in November 1585 then across the Atlantic at the Spanish new world city of Santo Domingo of which was captured and ransomed on 1 January 1586 and following that successfully attacked the important city of Cartagena on 19 February. Drake wanted to strike at another Spanish city on the Main before finally visiting and replenishing Sir Walter Raleigh’s new colony of Roanoke Colony on the American East Coast. Then after this he hoped to make the Transatlantic crossing back to England. The fleet headed north, and in late April Drake put into the Spanish Cuban mainland and his men dug wells in search of fresh water and gathered supplies to help counter an outbreak of dysentery after which he moved on. The fleet traveled north within sight of land on the Florida peninsula sailing past the West coast. On 27 May 1586 as they approached further north a small fort was spotted on the shore, with a small inlet close by. This was the location of St Augustine, the most northerly town in Spain’s New World Empire, and the oldest permanent colonial settlement in North America. Drake knew of the place and was also aware of the fact that the spanish under Pedro Menéndez de Avilés had ordered all of the French Huguenot colonists that had tried to settle in the area executed. Drake decided on one final opportunity to raid and plunder, and a chance to avenge his fellow Protestants.

Generate a summary about Lakewood (Livingston, Alabama) including the following aspects: architecture, history with one aspect per line.
# Architecture
The house has a plan that is relatively rare in early Alabama architecture. The plan features a brick ground floor that is topped by one-and-a-half-stories of wood-frame construction. The ground floor originally contained domestic spaces, with the formal rooms on the principle floor and bedrooms on the upper floor. A central hallway is present on all levels. The facade is five bays wide, with central entrance doors on the ground and principle floors. The bays are divided by two-story Doric pilasters, with the middle third of the facade occupied by a two-tiered tetrastyle Doric portico. Two curved wrought iron staircases ascend from ground level to the front center of the upper portico, leading to the formal entrance.
# History
Lakewood was built for Joseph lake, a native of North Carolina, by Hiram W. Bardwell, a master builder. Construction was completed in 1840. Located adjacent to the University of West Alabama, Julia Strudwick Tutwiler, a Lake relative, periodically resided in the house from 1881 to 1910 while she served as president of the university. It was then known as Livingston Normal College. The house was extensively photographed by Alex Bush for the Historic American Buildings Survey in November and December 1936. Lakewood has continued to be owned by descendants of the Lake family to the current day. The house and its surviving 10 acres (4.0 ha) of grounds were listed on the Places in Peril in 2012 due to the immediate threat of its acquisition by developers.

Generate a summary about Carlos Moedas including the following aspects: biography, early life, political career with one aspect per line.
```


---

## References


- **Borgeaud et al. (2022)** Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. [Improving language models by retrieving from trillions of tokens](https://proceedings.mlr.press/v162/borgeaud22a.html). In *International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA*, volume 162 of *Proceedings of Machine Learning Research*, pages 2206–2240. PMLR.

- **Brown et al. (2020)** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. [Language models are few-shot learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html). In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*.

- **Chen et al. (2017)** Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. [Reading wikipedia to answer open-domain questions](https://doi.org/10.18653/v1/P17-1171). In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers*, pages 1870–1879. Association for Computational Linguistics.

- **Chowdhery et al. (2022)** Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. [Palm: Scaling language modeling with pathways](https://doi.org/10.48550/arXiv.2204.02311). *CoRR*, abs/2204.02311.

- **Cohen et al. (2021)** Nachshon Cohen, Oren Kalinsky, Yftah Ziser, and Alessandro Moschitti. 2021. [Wikisum: Coherent summarization dataset for efficient human-evaluation](https://doi.org/10.18653/v1/2021.acl-short.28). In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 2: Short Papers), Virtual Event, August 1-6, 2021*, pages 212–219. Association for Computational Linguistics.

- **Dinan et al. (2019)** Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019. [Wizard of wikipedia: Knowledge-powered conversational agents](https://openreview.net/forum?id=r1l73iRqKm). In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net.

- **Fan et al. (2019)** Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. [ELI5: long form question answering](https://doi.org/10.18653/v1/p19-1346). In *Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers*, pages 3558–3567. Association for Computational Linguistics.

- **Gao et al. (2022)** Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022. [Precise zero-shot dense retrieval without relevance labels](https://doi.org/10.48550/arXiv.2212.10496). *CoRR*, abs/2212.10496.

- **Geva et al. (2021)** Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. *Transactions of the Association for Computational Linguistics*, 9:346–361.

- **Giorgi et al. (2022)** John M. Giorgi, Luca Soldaini, Bo Wang, Gary D. Bader, Kyle Lo, Lucy Lu Wang, and Arman Cohan. 2022. [Exploring the challenges of open domain multi-document summarization](https://doi.org/10.48550/arXiv.2212.10526). *CoRR*, abs/2212.10526.

- **Guu et al. (2020)** Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. [REALM: retrieval-augmented language model pre-training](http://arxiv.org/abs/2002.08909). *CoRR*, abs/2002.08909.

- **Hayashi et al. (2021)** Hiroaki Hayashi, Prashant Budania, Peng Wang, Chris Ackerson, Raj Neervannan, and Graham Neubig. 2021. [Wikiasp: A dataset for multi-domain aspect-based summarization](https://doi.org/10.1162/tacl_a_00362). *Trans. Assoc. Comput. Linguistics*, 9:211–225.

- **Hendrycks et al. (2020)** Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. [Measuring massive multitask language understanding](http://arxiv.org/abs/2009.03300). *CoRR*, abs/2009.03300.

- **Ho et al. (2020)** Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. [Constructing A multi-hop QA dataset for comprehensive evaluation of reasoning steps](https://doi.org/10.18653/v1/2020.coling-main.580). In *Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020*, pages 6609–6625. International Committee on Computational Linguistics.

- **Izacard and Grave (2021)** Gautier Izacard and Edouard Grave. 2021. [Leveraging passage retrieval with generative models for open domain question answering](https://doi.org/10.18653/v1/2021.eacl-main.74). In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021*, pages 874–880. Association for Computational Linguistics.

- **Izacard et al. (2022)** Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. [Few-shot learning with retrieval augmented language models](https://doi.org/10.48550/arXiv.2208.03299). *CoRR*, abs/2208.03299.

- **Jiang et al. (2021)** Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. 2021. [How can we know *When* language models know? on the calibration of language models for question answering](https://doi.org/10.1162/tacl_a_00407). *Trans. Assoc. Comput. Linguistics*, 9:962–977.

- **Jiang et al. (2022)** Zhengbao Jiang, Luyu Gao, Jun Araki, Haibo Ding, Zhiruo Wang, Jamie Callan, and Graham Neubig. 2022. [Retrieval as attention: End-to-end learning of retrieval and reading within a single transformer](https://doi.org/10.48550/arXiv.2212.02027). *CoRR*, abs/2212.02027.

- **Jiang et al. (2020)** Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. 2020. [How can we know what language models know](https://doi.org/10.1162/tacl_a_00324). *Trans. Assoc. Comput. Linguistics*, 8:423–438.

- **Joshi et al. (2017)** Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. [Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension](https://doi.org/10.18653/v1/P17-1147). In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers*, pages 1601–1611. Association for Computational Linguistics.

- **Kadavath et al. (2022)** Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. 2022. [Language models (mostly) know what they know](https://doi.org/10.48550/arXiv.2207.05221). *CoRR*, abs/2207.05221.

- **Karpukhin et al. (2020)** Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. [Dense passage retrieval for open-domain question answering](https://doi.org/10.18653/v1/2020.emnlp-main.550). In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pages 6769–6781. Association for Computational Linguistics.

- **Khandelwal et al. (2020)** Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020. [Generalization through memorization: Nearest neighbor language models](https://openreview.net/forum?id=HklBjCEKvH). In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net.

- **Khattab et al. (2022)** Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. [Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive NLP](https://doi.org/10.48550/arXiv.2212.14024). *CoRR*, abs/2212.14024.

- **Khot et al. (2022)** Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2022. [Decomposed prompting: A modular approach for solving complex tasks](https://doi.org/10.48550/arXiv.2210.02406). *CoRR*, abs/2210.02406.

- **Krishna et al. (2021)** Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. 2021. Hurdles to progress in long-form question answering. In *North American Association for Computational Linguistics*.

- **Kwiatkowski et al. (2019)** Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. [Natural questions: a benchmark for question answering research](https://doi.org/10.1162/tacl_a_00276). *Trans. Assoc. Comput. Linguistics*, 7:452–466.

- **Lazaridou et al. (2022)** Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. 2022. [Internet-augmented language models through few-shot prompting for open-domain question answering](https://doi.org/10.48550/arXiv.2203.05115). *CoRR*, abs/2203.05115.

- **Lee et al. (2021)** Haejun Lee, Akhil Kedia, Jongwon Lee, Ashwin Paranjape, Christopher D. Manning, and Kyoung-Gu Woo. 2021. [You only need one model for open-domain question answering](http://arxiv.org/abs/2112.07381). *CoRR*, abs/2112.07381.

- **Lewis et al. (2020)** Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. [Retrieval-augmented generation for knowledge-intensive NLP tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html). In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*.

- **Li et al. (2023)** Junyi Li, Tianyi Tang, Wayne Xin Zhao, Jingyuan Wang, Jian-Yun Nie, and Ji-Rong Wen. 2023. [The web can be your oyster for improving large language models](https://doi.org/10.48550/arXiv.2305.10998). *CoRR*, abs/2305.10998.

- **Lin (2004)** Chin-Yew Lin. 2004. [ROUGE: A package for automatic evaluation of summaries](https://aclanthology.org/W04-1013). In *Text Summarization Branches Out*, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

- **Liu et al. (2023)** Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. [Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing](https://doi.org/10.1145/3560815). *ACM Comput. Surv.*, 55(9):195:1–195:35.

- **Mallen et al. (2022)** Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. 2022. [When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories](https://doi.org/10.48550/arXiv.2212.10511). *CoRR*, abs/2212.10511.

- **Mao et al. (2021)** Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen. 2021. [Generation-augmented retrieval for open-domain question answering](https://doi.org/10.18653/v1/2021.acl-long.316). In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pages 4089–4100. Association for Computational Linguistics.

- **Maynez et al. (2020)** Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. [On faithfulness and factuality in abstractive summarization](https://doi.org/10.18653/v1/2020.acl-main.173). In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 1906–1919, Online. Association for Computational Linguistics.

- **Nakano et al. (2021)** Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. 2021. [Webgpt: Browser-assisted question-answering with human feedback](http://arxiv.org/abs/2112.09332). *CoRR*, abs/2112.09332.

- **OpenAI (2023)** OpenAI. 2023. [GPT-4 technical report](https://doi.org/10.48550/arXiv.2303.08774). *CoRR*, abs/2303.08774.

- **Ouyang et al. (2022)** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. [Training language models to follow instructions with human feedback](https://doi.org/10.48550/arXiv.2203.02155). *CoRR*, abs/2203.02155.

- **Peng et al. (2023)** Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. 2023. [Check your facts and try again: Improving large language models with external knowledge and automated feedback](https://doi.org/10.48550/arXiv.2302.12813). *CoRR*, abs/2302.12813.

- **Petroni et al. (2019)** Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick S. H. Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander H. Miller. 2019. [Language models as knowledge bases?](https://doi.org/10.18653/v1/D19-1250) In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019*, pages 2463–2473. Association for Computational Linguistics.

- **Press et al. (2022)** Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2022. Measuring and narrowing the compositionality gap in language models. *arXiv preprint arXiv:2210.03350*.

- **Qian et al. (2023)** Hongjing Qian, Yutao Zhu, Zhicheng Dou, Haoqi Gu, Xinyu Zhang, Zheng Liu, Ruofei Lai, Zhao Cao, Jian-Yun Nie, and Ji-Rong Wen. 2023. [Webbrain: Learning to generate factually correct articles for queries by grounding on large web corpus](https://doi.org/10.48550/arXiv.2304.04358). *CoRR*, abs/2304.04358.

- **Qin et al. (2023)** Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Lin, Xu Han, Ning Ding, Huadong Wang, Ruobing Xie, Fanchao Qi, Zhiyuan Liu, Maosong Sun, and Jie Zhou. 2023. [Webcpm: Interactive web search for chinese long-form question answering](https://doi.org/10.48550/arXiv.2305.06849). *CoRR*, abs/2305.06849.

- **Radford et al. (2019)** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). *OpenAI Blog*, 1(8).

- **Ram et al. (2023)** Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language models. *arXiv preprint arXiv:2302.00083*.

- **Roberts et al. (2020)** Adam Roberts, Colin Raffel, and Noam Shazeer. 2020. [How much knowledge can you pack into the parameters of a language model?](https://doi.org/10.18653/v1/2020.emnlp-main.437) In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pages 5418–5426. Association for Computational Linguistics.

- **Robertson and Zaragoza (2009)** Stephen E. Robertson and Hugo Zaragoza. 2009. [The probabilistic relevance framework: BM25 and beyond](https://doi.org/10.1561/1500000019). *Found. Trends Inf. Retr.*, 3(4):333–389.

- **Sachan et al. (2021)** Devendra Singh Sachan, Siva Reddy, William L. Hamilton, Chris Dyer, and Dani Yogatama. 2021. [End-to-end training of multi-document reader and retriever for open-domain question answering](https://proceedings.neurips.cc/paper/2021/hash/da3fde159d754a2555eaa198d2d105b2-Abstract.html). In *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pages 25968–25981.

- **Schick et al. (2023)** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. [Toolformer: Language models can teach themselves to use tools](http://arxiv.org/abs/2302.04761).

- **Shi et al. (2023)** Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. [REPLUG: retrieval-augmented black-box language models](https://doi.org/10.48550/arXiv.2301.12652). *CoRR*, abs/2301.12652.

- **Stelmakh et al. (2022)** Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. 2022. [ASQA: factoid questions meet long-form answers](https://aclanthology.org/2022.emnlp-main.566). In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pages 8273–8288. Association for Computational Linguistics.

- **Sun et al. (2022)** Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. 2022. [Recitation-augmented language models](https://doi.org/10.48550/arXiv.2210.01296). *CoRR*, abs/2210.01296.

- **Touvron et al. (2023)** Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. [Llama: Open and efficient foundation language models](https://doi.org/10.48550/arXiv.2302.13971). *CoRR*, abs/2302.13971.

- **Trivedi et al. (2022)** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. [Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions](https://doi.org/10.48550/arXiv.2212.10509). *CoRR*, abs/2212.10509.

- **Varshney et al. (2022)** Neeraj Varshney, Man Luo, and Chitta Baral. 2022. [Can open-domain QA reader utilize external knowledge efficiently like humans?](https://doi.org/10.48550/arXiv.2211.12707) *CoRR*, abs/2211.12707.

- **Wang et al. (2022)** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, and Denny Zhou. 2022. [Self-consistency improves chain of thought reasoning in language models](https://doi.org/10.48550/arXiv.2203.11171). *CoRR*, abs/2203.11171.

- **Wei et al. (2022)** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. 2022. [Chain of thought prompting elicits reasoning in large language models](http://arxiv.org/abs/2201.11903). *CoRR*, abs/2201.11903.

- **Yao et al. (2022)** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. [React: Synergizing reasoning and acting in language models](https://doi.org/10.48550/arXiv.2210.03629). *CoRR*, abs/2210.03629.

- **Yu et al. (2022)** Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2022. [Generate rather than retrieve: Large language models are strong context generators](https://doi.org/10.48550/arXiv.2209.10063). *CoRR*, abs/2209.10063.

- **Yu et al. (2023)** Wenhao Yu, Zhihan Zhang, Zhenwen Liang, Meng Jiang, and Ashish Sabharwal. 2023. [Improving language models via plug-and-play retrieval feedback](https://doi.org/10.48550/arXiv.2305.14002). *CoRR*, abs/2305.14002.

- **Zemlyanskiy et al. (2022)** Yury Zemlyanskiy, Michiel de Jong, Joshua Ainslie, Panupong Pasupat, Peter Shaw, Linlu Qiu, Sumit Sanghai, and Fei Sha. 2022. [Generate-and-retrieve: Use your predictions to improve retrieval for semantic parsing](https://aclanthology.org/2022.coling-1.438). In *Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022*, pages 4946–4951. International Committee on Computational Linguistics.

- **Zhang et al. (2023)** Fengji Zhang, Bei Chen, Yue Zhang, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. [Repocoder: Repository-level code completion through iterative retrieval and generation](https://doi.org/10.48550/arXiv.2303.12570). *CoRR*, abs/2303.12570.

- **Zhang et al. (2022)** Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pre-trained transformer language models. *ArXiv*, abs/2205.01068.

- **Zhao et al. (2023)** Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. [A survey of large language models](https://doi.org/10.48550/arXiv.2303.18223). *CoRR*, abs/2303.18223.

- **Zhong et al. (2022)** Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. 2022. [Towards a unified multi-dimensional evaluator for text generation](https://aclanthology.org/2022.emnlp-main.131). In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pages 2023–2038. Association for Computational Linguistics.

- **Zhou et al. (2021)** Chunting Zhou, Graham Neubig, Jiatao Gu, Mona Diab, Francisco Guzmán, Luke Zettlemoyer, and Marjan Ghazvininejad. 2021. [Detecting hallucinated content in conditional neural sequence generation](https://doi.org/10.18653/v1/2021.findings-acl.120). In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 1393–1404, Online. Association for Computational Linguistics.
