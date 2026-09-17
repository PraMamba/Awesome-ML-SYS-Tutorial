# Polar: A<sub>g</sub>entic RL on An<sub>y</sub> Harness at Scale

Binfeng Xu, Hao Zhang, Shaokun Zhang, Songyang Han, Mingjie Liu, Jian Hu, Shizhe Diao, Zhenghui Jin, Yunhen<sub>g</sub> Zou<sub>,</sub> Michael Demoret<sub>,</sub> Jan Kautz<sub>,</sub> Yi Don<sub>g</sub>

## Ab<sub>s</sub>tr<sub>ac</sub>t

R<sub>e</sub>i<sub>n</sub>f<sub>orcemen</sub>t l<sub>earn</sub>i<sub>ng</sub> f<sub>or</sub> l<sub>anguage agen</sub>t<sub>s</sub> i<sub>ncreas</sub>i<sub>ng</sub>l<sub>y</sub> d<sub>epen</sub>d<sub>s on cus</sub>t<sub>om</sub> h<sub>arnesses</sub> th<sub>a</sub>t <sub>manage</sub> l<sub>ong-</sub> <sub>runn</sub>i<sub>n con</sub>t<sub>ex</sub>t<sub>, mu</sub>lti<sub>-</sub>t<sub>urn</sub> t<sub>oo</sub>l <sub>use an</sub>d <sub>mu</sub>lti<sub>-a en</sub>t <sub>orc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on.</sub> H<sub>owever, or</sub>ti<sub>n</sub> th<sub>ese</sub> h<sub>arnesses</sub> i<sub>n</sub>t<sub>o</sub> RL <sub>env</sub>i<sub>ronmen</sub>t i<sub>n</sub>t<sub>er</sub>f<sub>aces rema</sub>i<sub>ns</sub> difi<sub>cu</sub>lt <sub>an</sub>d <sub>o</sub>ft<sub>en</sub> l<sub>oses</sub> i<sub>m or</sub>t<sub>an</sub>t t<sub>ra</sub>i<sub>n</sub>i<sub>n s</sub>i <sub>na</sub>l<sub>s.</sub> W<sub>e</sub> b<sub>r</sub>id <sub>e</sub> thi<sub>s gap w</sub>ith P<sub>olar, a ro</sub>ll<sub>ou</sub>t f<sub>ramewor</sub>k f<sub>or sca</sub>l<sub>a</sub>bl<sub>e async</sub>h<sub>ronous</sub> RL <sub>over ar</sub>bit<sub>rary agen</sub>t h<sub>arnesses.</sub> P<sub>olar</sub> t<sub>rea</sub>t<sub>s</sub> th<sub>e a en</sub>t h<sub>arness as a</sub> bl<sub>ac</sub>k b<sub>ox:</sub> it <sub>rox</sub>i<sub>es</sub> LLM API <sub>ca</sub>ll<sub>s, recor</sub>d<sub>s</sub> t<sub>o</sub>k<sub>en-</sub>l<sub>eve</sub>l <sub>mo</sub>d<sub>e</sub>l interactions, and reconstructs token-faithful trajectories for training. Each rollout node eficiently manages runtime prewarming, agent execution, trajectory reconstruction, and evaluation in parallel, <sub>expos</sub>i<sub>ng async</sub>h<sub>ronous serv</sub>i<sub>ce en</sub>d<sub>po</sub>i<sub>n</sub>t<sub>s</sub> th<sub>a</sub>t <sub>can</sub> b<sub>e consume</sub>d b<sub>y</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t t<sub>ra</sub>i<sub>ners a</sub>t <sub>sca</sub>l<sub>e.</sub> Thi<sub>s</sub> d<sub>ecoup</sub>l<sub>e</sub>d d<sub>es</sub>i<sub>gn ma</sub>k<sub>es</sub> P<sub>olar agnos</sub>ti<sub>c</sub> t<sub>o agen</sub>t h<sub>arnesses,</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> i<sub>n</sub>f<sub>ras</sub>t<sub>ruc</sub>t<sub>ure, an</sub>d RL <sub>a</sub>l<sub>gor</sub>ith<sub>ms</sub> <sub>w</sub>hil<sub>e</sub> i<sub>mprov</sub>i<sub>ng compu</sub>t<sub>e u</sub>tili<sub>za</sub>ti<sub>on</sub> f<sub>or</sub> l<sub>ong-runn</sub>i<sub>ng agen</sub>t <sub>wor</sub>kl<sub>oa</sub>d<sub>s.</sub> W<sub>e va</sub>lid<sub>a</sub>t<sub>e</sub> P<sub>olar</sub> b<sub>y</sub> t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> <sub>agen</sub>t<sub>s on so</sub>ft<sub>ware-eng</sub>i<sub>neer</sub>i<sub>ng</sub> t<sub>as</sub>k<sub>s w</sub>ith <sub>popu</sub>l<sub>ar co</sub>di<sub>ng</sub> h<sub>arnesses.</sub> U<sub>s</sub>i<sub>ng s</sub>i<sub>mp</sub>l<sub>e</sub> GRPO<sub>,</sub> P<sub>olar</sub> im roves Qwen3.5-4B b 22.6 4.8 0.6 and 6.2 oints on SWE-Bench Verified with the Codex Claude Code, Qwen Code and Pi harnesses, respectively. We further demonstrate Polar for ofline data generation over custom harnesses and ablate trajectory reconstruction strategies. Polar rewrites its <sub>prece</sub>di<sub>ng wor</sub>k<sub>,</sub> P<sub>ro</sub>RL A<sub>gent</sub><sup>1</sup> <sub>an</sub>d h<sub>as</sub> b<sub>een reg</sub>i<sub>s</sub>t<sub>ere</sub>d <sub>as one o</sub>f N<sub>e</sub>M<sub>o</sub> G<sub>ym env</sub>i<sub>ronmen</sub>t<sub>s.</sub>

![](images/2bc40857936fcabd8fd6c5261cec6f2cfb9e9d7d73c45c7b199f5d10029f90b0.jpg)  
Figure 1: Polar arc<sup>h</sup>itecture overview. Polar runs an existing agent harness inside an isolated runtime and places a model API proxy between the harness and the inference server. The proxy forwards model calls, records token-level request and response data, and reconstructs RL trajectories, while rollout gateways asynchronously handle runtime prewarming, harness execution, evaluation, and trainer callbacks. This decoupled design allows Polar to treat agents as a black-box environment, seamlessly scaling across diferent training frameworks.

![](images/91a4f648f3333d63a37477c65a39e0570a7cc10b3c23e7b934568b5c60f93e6c.jpg)  
Figure 2: Polar uses t<sup>h</sup>e mo<sup>d</sup>e<sup>l</sup> API proxy as t<sup>h</sup>e ro<sup>ll</sup>out <sup>b</sup>oun<sup>d</sup>ary. Traditional rollout frameworks usually require the agent or harness logic to be rewritten behind a framework-owned environment API. This makes the trainer depend on harness-specific integration code and can miss details of the native execution path. Polar instead keeps the harness unchanged and places a provider-compatible proxy at the LLM API boundary; the proxy records prompts, sampled tokens, log probabilities, and responses, then reconstructs trainer-ready trajectories outside the harness.

## 1<sub>.</sub> I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Reinforcement learning for large language model is moving beyond short, single-step tasks toward agentic settings (Tan et al., 2025; Zhang et al., 2026a) that require sustained interaction with external environments, such as code repositories (Jimenez et al., 2024; Pan et al., 2024), web browsers (Deng et al., 2023; Zhou et al., 2023), and even full operating systems (Wang et al., 2025; Xie et al., 2024), through iterative tool use (Guo et al., 2025; Patil et al., 2025; Shao et al., 2024). These settings often produce long-horizon trajectories with dozens of interaction steps and tens of thousands of tokens.

This shift makes the training target itself a central systems challenge for agentic RL. Traditional RL often assumes the training target can be exposed through a simple, standardized interface (Brockman et al., 2016), allowing researchers to focus mainly on the RL algorithm. In agentic RL, however, the training target is often a complex software system (Anthropic, 2026; OpenAI, 2026). It may involve heterogeneous environments (NVIDIA, 2025), various external tools (Zhang et al., 2024, 2026b), and long-running workflows (Jimenez et al., 2024), and may be implemented in diferent languages or even distributed as a closed-source binary (Anthropic, 2026).

This creates substantial integration burdens. For example, in building agentic RL systems, SkyRL-Agent (Cao et al., 2025) and PRIME-RL (Prime Intellect, 2026) integrate agent execution directly into the RL pipeline, requiring users to adapt their agents to the RL infrastructure rather than allowing the infrastructure to accommodate existing agent implementations. This makes the design less flexible: every new agent or harness often requires one framework-specific integration. Some recent systems attempt to make this integration less intrusive. For instance, Agent Lightning (Luo et al., 2025) and rLLM (Tan et al., 2025) reduce this burden by introducing standard tracing interfaces and LLM-call capture mechanisms, but still require agents to conform to prescribed interfaces. Thus, these systems lower the cost of integration but do not fully eliminate it. These issues are likely to become more severe as agent harnesses grow increasingly complex and, in some cases, even do not expose their internal implementation, making conventional RL integration dificult or even infeasible. Motivated by these issues, we explore the following central question:

## Can we train agents with RL without opening the box?

That is, without touching their harnesses or forcing them to conform to an RL framework. The key observation is that, although agents difer widely in their internal implementations, every LLM-based agent must talk to a model. This model API boundary provides a common interface that exists outside the agent itself. Instead of integrating with the agent harness, we can train by listening to the agent’s LLM calls: capturing its prompts, sampled tokens, log probabilities, and responses, and converting them into RL trajectories. In this view, an agent can be treated as a black box while still becoming trainable.

Building on this intuition, we present Polar: an agentic RL infrastructure that could train any agents as black boxes. The name Polar reflects both its roots in PrOrL Agent servR (Zhang et al., 2026a) and its role in connecting the two “poles” of agent training and deployment: the training environment and the product harness. Instead of treating the agent harness as the RL interface, Polar uses the agent’s LLM API trafic as the interface. Through listening to its model calls through a proxy and converts them into trajectories and rewards for training, the agent runs unchanged.

In addition, Polar separates runtime setup, agent execution, trajectory reconstruction, evaluation, and trainer callbacks behind asynchronous service boundaries. This allows slow and long-tail agent rollouts to scale independently from GPU training, exposing a trainer-agnostic rollout-as-a-service interface for scaling eficient RL infrastructures (Zhang et al., 2026a). In summary, the main contributions of this work are:

• Proxy <sup>b</sup>ase<sup>d</sup> ro<sup>ll</sup>out an<sup>d</sup> reconstruction over agent <sup>h</sup>arnesses. We propose a paradigm using the agent’s LLM API payloads as the RL rollout interface, allowing existing harnesses to serve directly as RL environments without internal code change.

• Ro<sup>ll</sup>out-as-a-service arc<sup>h</sup>itecture <sup>f</sup>or sca<sup>l</sup>ing RL in<sup>f</sup>rastructures. Polar separates task submission, runtime setup, harness execution, trajectory reconstruction, evaluation, and trainer callbacks behind asynchronous service boundaries, natively scaling with modern RL infrastructures.

• To<sup>k</sup>en-<sup>f</sup>ait<sup>hf</sup>u<sup>l</sup> trajectory reconstruction. Polar converts raw model requests into token-faithful traces for training. We provide conservative per-request reconstruction and prefix merging for heavy rolllouts, while leaving registry-based extensible interfaces.

• En<sup>d</sup>-to-en<sup>d</sup> va<sup>l</sup>i<sup>d</sup>ation on rea<sup>l</sup>-wor<sup>ld</sup> co<sup>d</sup>ing <sup>h</sup>arnesses. We validate Polar with RL training on various popular harnesses for software-engineering tasks, and further demonstrate ofline SFT data generation with a custom coding harness.

## 2<sub>.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

A compact checklist of rollout-system design choices is provided in Tab. 3 in the appendix. This section focuses on the qualitative diferences behind that comparison.

## 2.1. A<sub>g</sub>ent RL S<sub>y</sub>stems

The first wave of LLM RL infrastructure largely assumed that rollout generation was a Python function owned by the trainer. This assumption is increasingly strained by multi-turn agents, where interaction spans many model calls and environment actions. ProRL Agent (Zhang et al., 2026a) introduced a service boundary for multi-turn agent rollouts, separating sandbox setup, agent execution, and reward computation from the training process. Polar inherits the same high-level idea that rollout should be a service, but changes the integration contract. Instead of implementing an agent handler inside the rollout service, the user supplies a harness adapter that prepares configuration and launches the native executable. The model proxy then observes the harness from outside.

SkyRL-Agent (Cao et al., 2025) is a full-stack system for eficient RL training and evaluation of multi-turn, long-horizon agents, with SkyRL-Gym providing tool-use environments through a Gymnasium-style interface. SkyRL’s strength is eficient training once tasks are represented in its environment and agent abstractions. Polar is complementary: it targets the earlier systems problem of running a pre-existing harness whose internal event loop, tool formatting, and context policy should remain unchanged.

PRIME-RL (Prime Intellect, 2026) focuses on large-scale asynchronous RL with trainer-inference separation, stale-policy step semantics, and support for verifiers environments. Slime (Zheng et al., 2024; Zhu et al., 2025) similarly connects Megatron training with SGLang rollout engines and exposes customizable data-generation interfaces. These systems address the policy-optimization and inference-scaling side of the pipeline. Polar is not a replacement trainer. It is a rollout substrate that can feed asynchronous trainers with trajectories from heavier harnesses than typical verifiable-reward functions.

## 2.2. Low-Intrusion A<sub>g</sub>ent Instrumentation

Agent Lightning (Luo et al., 2025) proposes a training-agent disaggregation architecture and a unified data interface for converting agent execution into trainable transitions. rLLM (Tan et al., 2025) similarly aims to train agents across frameworks with minimal code changes, using tracked clients, decorators, workflow abstractions, and proxy support to collect token IDs and log probabilities. Both systems recognize that researchers should not have to rewrite complete applications to train them.

Polar difers in the chosen minimum integration point. For many coding and terminal agents, the most reliable interface is not an SDK callback graph but the provider API endpoint already used by the harness. The gateway proxy therefore becomes the observation device: it accepts Anthropic, OpenAI Chat, OpenAI Responses, and Google-style requests; translates them to the local inference backend; and records the token-level fields needed by the trainer. This choice is narrower than general observability instrumentation, but it is robust to harnesses implemented as command-line programs, package-managed tools, or binaries.

## 2<sub>.</sub>3<sub>.</sub> SWE T<sub>as</sub>k E<sub>va</sub>l<sub>ua</sub>ti<sub>on an</sub>d B<sub>enc</sub>h<sub>mar</sub>k

Harbor (Harbor Framework Team, 2026) evaluates agents such as Claude Code, OpenHands, Codex CLI, and related systems in containerized environments, supports parallel execution through local and cloud providers, and converts native agent logs into evaluation trajectories. This evaluation-first design is highly aligned with Polar’s harness-native motivation. The diference is the model boundary and training data contract. Harbor launches each harness with provider-specific configuration and does not provide a gateway that translates model-provider protocols or mediates the harness’s model trafic. As a result, model substitution is limited by what the native harness and external endpoint already support: for example, evaluating a Qwen checkpoint through Claude Code requires an Anthropic-compatible endpoint outside Harbor. Polar instead places a proxy at this boundary, so the same style of harness execution can yield token IDs, log probabilities, loss masks, and rewards that are directly consumable by an RL trainer.

SWE-bench (Jimenez et al., 2024) established real GitHub issue resolution as a benchmark requiring repository understanding, editing, and executable validation. SWE-Gym (Pan et al., 2024) extends this direction with training environments, verifiers, and trajectories for software-engineering agents. These workloads are a natural stress test for rollout infrastructure because they combine expensive runtime setup, sparse patch-level rewards, long-tail execution time, and many opportunities for harness-side state to diverge from a clean evaluator state.

## 2<sub>.</sub>4<sub>.</sub> T<sub>o</sub>k<sub>en</sub> Fid<sub>e</sub>lit <sub>an</sub>d R<sub>e</sub>t<sub>o</sub>k<sub>en</sub>i<sub>za</sub>ti<sub>on</sub> D<sub>r</sub>ift

The training signal in agent RL is only correct if it is attached to the tokens sampled by the behavior policy. This is dificult in agent harnesses because provider APIs may return text, tool-call JSON, reasoning fields, or d h h h k d l b b l d b h f b k d h and Agent Lightning discussion of retokenization drift emphasizes that decoding and re-encoding a transcript can produce diferent token IDs from the original generation (Agent Lightning Team, 2025). Polar follows the same token-fidelity principle but applies it to arbitrary harness rollouts: generated assistant tokens are copied from inference responses, non-generated interstitial tokens are taken from canonical prompt tokenization, and the loss mask marks only behavior-policy tokens as trainable.

## 3. Polar

We target agentic RL tasks where a policy is exercised through an existing harness rather than a custom rollout loop. A task starts from an instruction and a runtime; the harness calls a model endpoint while using tools, editing files, spawning sub-agents, or managing context (compaction, injection, replacement, etc.). After execution, an evaluator assigns an outcome or trace-level reward. The rollout system must preserve the native model interactions as trainer-ready traces: prompt context, sampled assistant tokens, optional behavior-policy log probabilities, loss masks, rewards, and provenance.

![](images/2b009ada633fe731036c8cefeda7790d1297cd5a49bce4837fca9f048a05cc7f.jpg)  
Figure 3: Gateway-<sup>l</sup>eve<sup>l</sup> async<sup>h</sup>ronous staging in Polar. A gateway separates runtime initialization, ready bufering, harness execution, and post-run trajectory and evaluation work into isolated worker pools. Runtime preparation and evaluator prewarm proceed of the critical path, so CPU-heavy runtime setup and long-tail evaluation do not block active GPU-bound agent run.

## 3.1. Architecture

Polar has two core components: a rollout server and gateway nodes. The rollout service coordinates tasks and global scheduling. Gateway nodes execute sessions, host the model proxy, construct trajectories, and run evaluation. This split keeps durable task management separate from per-session execution and capture.

## R<sub>o</sub>ll<sub>ou</sub>t <sub>server.</sub>

The rollout service accepts a TaskRequest and expands it into num\_samples independent sessions. A session is the scheduling unit: it has a session ID, task ID, timeout budget, runtime specification, agent specification, trajectory builder, evaluator, and callback URL. The service dispatches sessions to gateway nodes, persists compact terminal results, exposes task status through polling, and accepts gateway callbacks when sessions finish. Sec. A.3 gives a representative payload.

## Gatewa<sub>y</sub> node.

A gateway owns the lifecycle of each session. It starts the runtime, prepares the harness, runs the harness commands, builds trajectories from captured completions, evaluates the output, tears down resources, and returns the result. The same gateway also hosts the proxy endpoint used by the harness for model calls. This co-location keeps completion capture tied to the session registry and avoids a separate trace-collection service.

Training frameworks are independent from Polar servers. And the service boundaries natively supports eficient asynchronous RL at scale. Fig. 5a shows one such example with Slime: a background worker submits Polar tasks, receives task-completion callbacks, converts traces into Slime Sample objects, and applies trajectory-aware reward post-processing.

## 3.2. Harness and Prox<sub>y</sub> Ca<sub>p</sub>ture

Polar observes native harnesses by routing their model calls through the gateway proxy. A harness is configured through its normal environment variables or config files so that its model base URL points to the gateway.

For each incoming model request, the gateway performs four steps.

1. Detect t<sup>h</sup>e provi<sup>d</sup>er API. Detection uses the request path and headers to distinguish Anthropic Messages, OpenAI Chat Completions, OpenAI Responses, and Google generateContent-style calls.

2. Norma<sup>l</sup>ize t<sup>h</sup>e request. A provider transformer converts roles, content parts, tool definitions, tool choices, stop controls, and generation parameters into the OpenAI Chat Completions shape consumed by local inference servers. The transformer also adds fields needed for training, such as logprobs=true.

3. Capture to<sup>k</sup>en-<sup>l</sup>eve<sup>l d</sup>ata. The gateway forwards the normalized request to the inference servers and stores a completion record containing the request messages, response messages, prompt token IDs, sampled response token IDs, finish reason, and log probabilities from inference backends.

4. Return t<sup>h</sup>e provi<sup>d</sup>er s<sup>h</sup>ape. The response is transformed back to the schema expected by the harness. For streaming requests, our implementation obtains a non-streaming upstream response and emits a synthetic provider-shaped stream. This simplifies faithful token capture while preserving compatibility with harnesses that expect server-sent events.

The proxy boundary is intentionally below the agent framework. It does not need to understand how the harness plans, manages tools, or decides when to stop. It only needs to preserve API compatibility and record enough information to reconstruct training samples.

## 3.2.1. Harness Ada<sub>p</sub>ter

A harness adapter in Polar is small by design. It may install configuration, register MCP servers or skills, write provider settings, and return the shell commands that run the agent. A generic shell command harness can be used for wrapped agent execution. We also integrate popular agent harnesses as shortcuts like claude\_code, codex, gemini\_cli, qwen\_code, opencode and pi.

## 3.2.2. Runtime Interface

Runtimes implement a common interface for start, stop, exec, upload, download, and cancellation. Our first release supports Docker and rootless Apptainer for HPC setup. Because gateway code only depends on the runtime interface, a task can change isolation backend without friction.

## 3.3. As<sub>y</sub>nchronous Rollout Sta<sub>g</sub>in<sub>g</sub>

Long-horizon harness rollouts mix several diferent costs: runtime startup, dependency preparation, harness execution, evaluator setup, test execution, patch application, and teardown. Polar keeps these costs from blocking one another through stage-isolated execution inside each gateway (Fig. 3).

## 3.3.1. In-node worker <sub>p</sub>ools.

Each gateway uses isolated worker pools for INIT, RUNNING, and POSTRUN, plus a bounded READY bufer. INIT starts the runtime and executes prepare actions. READY holds initialized runtimes until a run slot is available. RUNNING executes the harness. POSTRUN builds trajectories, runs evaluators, executes post-run hooks, sends callbacks, and tears down resources. The ready bufer allows CPU-heavy runtime preparation to proceed in the background without blocking GPU-bound agent execution.

## 3.3.2. Evaluator <sub>p</sub>rewarm and timeouts.

When an evaluator requests a clean runtime, the gateway begins preparing that runtime during the agent run. Each session also carries one shared deadline; if a harness times out after model calls have been captured, the gateway still enters post-run so partial traces can be recovered with terminal timeout status.

![](images/1758fafe3d80f8ddf54c79269b386ae07470e15a00500a00585c5ae50147267a.jpg)  
Figure 4: Trajector reconstruction exam <sup>l</sup>e. The visualized session contains a three-turn main agent that undergoes one harness-level context compaction and spawns one subagent. The per-request builder keeps each captured model call as an independent trace. Prefix merging instead recovers append-only conversation chains where valid, while compaction and subagent boundaries naturally form separate chains. Within each merged trace, Polar prefix merging algorithm copies only sampled assistant tokens as trainable tokens and masks canonical interstitial tokens, preserving behavior-policy fidelity while reducing trainer-facing samples.

## 3.4. Trajectory Reconstruction

The trajectory builder interface converts an ordered CompletionSession into a Trajectory. A completion session is the stored sequence of proxy-captured model calls for one harness session. A trajectory contains one or more Trace objects, each with prompt token IDs, response token IDs, a loss mask, prompt messages, response messages, tool definitions, log probabilities, reward, and metadata. Sec. A.4 shows a representative trainer-facing trace. Custom trajectory strategies can be seamlessly added to the registry, and we provide two strategies, per request and prefix merging, compared in Fig. 4.

## 3.4.1. Per Re<sub>q</sub>uest

The per\_request builder is the conservative baseline as shown in the bottom left of Fig. 4: every completion becomes one trace. This is lossless with respect to individual calls, but it can fragment a coherent multi-turn agent session into many short samples. For complex coding harnesses, solving a single coding problem can produce hundreds of such traces, which increases the burden on downstream trainers.

## 3.4.2. Token Faithful Prefix Mer<sub>g</sub>in<sub>g</sub>

The prefix\_merging builder reconstructs longer traces when parts of a harness session preserve append-only conversation histories as shown in the bottom right of Fig. 4. It does not assume that the whole session is a single conversation. Instead, for a session with completions $C _ { 1 } , \ldots , C _ { T }$ , where completion $C _ { i }$ has prompt token sequence $p _ { i }$ , raw sampled response token sequence $a _ { i }$ , response log probabilities $\ell _ { i }$ , and prompt/response messages $m _ { i }$ , Polar partitions the completions into ordered chains

$$
\mathcal {G} = \{G _ {1}, \dots , G _ {J} \}, \qquad G _ {j} = (C _ {i _ {1} ^ {j}}, C _ {i _ {2} ^ {j}}, \dots , C _ {i _ {K _ {j}} ^ {j}}),
$$

with $i _ { 1 } ^ { j } < i _ { 2 } ^ { j } < \cdots < i _ { K _ { j } } ^ { j }$ . A new completion can join an existing chain only when a normalized message-level grouping key identifies it as a candidate continuation and the strict token-prefix relation holds against the last prompt in that chain. For adjacent completions $C _ { i _ { m } }$ and $C _ { i _ { m + 1 } }$ inside one chain, this check is

$$
p _ {i _ {m + 1}} [ 1: | p _ {i _ {m}} | ] = p _ {i _ {m}}.
$$

![](images/8fdc46404bf3cd154ea2d8a34ad887de9a1f85f4fb05e3ce49d0b406861e814b.jpg)  
(a) Async RL.

![](images/124a8598dbc51ee2de56497c60bd005d74d8b942b5fffc0757ffee73a64efe82.jpg)  
(b) GPU utilization with diferent reconstruction strategies.  
Figure 5: Polar improves GPU uti<sup>l</sup>ization across t<sup>h</sup>e ro<sup>ll</sup>out-training <sup>b</sup>oun<sup>d</sup>ary. (a) Shows an asynchronous RL pipeline enabled by Polar services. The rollout server keeps inferencing with existing policy, while trainer steps only if receiving batch size of evaluated trajectory groups. (b) Shows a span of 3 training steps under the same workload and topology, prefix merging emits fewer trainer updates than per-request reconstruction and substantially accelerates the training process.

Thus sub-agents, parallel agent branches, context compaction, prompt rewriting, or independent tool-mediated conversations naturally form additional chains rather than being forced into one global trace.

Merging is then applied independently to each chain. Consider one chain $G = ( C _ { i _ { 1 } } , \dots , C _ { i _ { K } } )$ and write $p _ { m } = p _ { i _ { m } } , a _ { m } = a _ { i _ { m } }$ , and $\ell _ { m } = \ell _ { i _ { m } }$ . The main challenge is that $p _ { m + 1 }$ contains a canonical server rendering of the previous assistant turn plus the interstitial context inserted by the harness before the next generation prompt. The previous assistant body must not be copied from this canonical rendering, because the behavior-policy tokens are the raw sampled tokens $a _ { m }$ . Let � denote the end-of-turn token ID. For two adjacent completions in the chain, define the canonical tail

$$
t _ {m} = p _ {m + 1} [ | p _ {m} | + 1: ].
$$

We locates the first � in $t _ { m } . \mathrm { ~ I f ~ } a _ { m }$ already ends with $e ,$ the interstitial $u _ { m }$ is the sufix after that $e ;$ otherwise $u _ { m }$ starts at that � so the assistant turn is still closed before the next prompt context. The token sequence represented by this chain is

$$
z ^ {(j)} = p _ {1} \mid \mid a _ {1} \mid \mid u _ {1} \mid \mid a _ {2} \mid \mid u _ {2} \mid \mid \dots \mid \mid a _ {K}.
$$

The emitted trajectory therefore contains one trace $\tau ^ { ( j ) }$ per chain, with the first prompt $p _ { 1 }$ stored as the trace prompt and the remaining sufix $a _ { 1 } { \left| \left| u _ { 1 } \right| \right| } \cdots { \left| \left| a _ { K } \right| \right| }$ stored as the trace response. The explicit loss mask is one on tokens copied from sampled responses $a _ { m }$ and zero on tokens copied from canonical interstitials $u _ { m }$ . Real response log-probability entries are copied for $a _ { m }$ tokens. Interstitial slots receive synthetic log-probability entries so stays aligned with ; trainability is controlled by .

This construction gives a simple correctness invariant in every emitted trace:

Every trainable token matches the behavior policy during rollout, and any non-generated tokens are masked out.

![](images/73a4c79ab820ca4c203d7ffe60d13a4ef584a6ee610969c3a6cea618c8341ba6.jpg)  
Figure 6: SWE-Gym GRPO training curves. Each panel shows the per-step outcome reward, equivalent to rollout pass@1, for one of four evaluated coding harnesses. RL improves reward across harnesses, with the clear gains on execution paths involving complex prompting, orchestration, or unfamiliar tool schemas.

Fig. 5b compares the GPU utilizations of the 2 strategies above with the same configurations.

## 3.5. Evaluation and Reward Pro<sub>p</sub>a<sub>g</sub>ation

Evaluators are registry-backed custom strategies that run after trajectory construction. They receive the trajectory, session artifacts, and optionally refreshed runtime context. Built-in evaluators include a sessioncompletion reward, a configurable test-on-output evaluator, and a SWE-Bench/SWE-Gym harness evaluator. An outcome reward can be broadcast to every trace, whereas tasks with process rewards may need pertrace assignment. The evaluator registry allows straightforward extension to custom rule-based verification, agent-as-judge scoring, and task-specific reward shaping.

## 4. Ex<sub>p</sub>eriments

We validate Polar in two settings: online RL rollout and ofline SFT data generation. The experiments test whether unchanged harnesses can produce trainable traces for both reward-driven and supervised training.

## 4.1. SWE-G<sub>y</sub>m GRPO on Codin<sub>g</sub> Harnesses

We run standard GRPO over four representative coding harnesses with PoLAR. Starting from the same Owen3.5- 4B base checkpoint, we run standard GRPO training on the SkyRL-v0-293-data SWE-Gym dataset,<sup>2</sup> with Polar and Slime. We use the training split for policy optimization and reserve evaluation for SWE-Bench Verified (Jimenez et al., 2024). All experiments use prefix\_merging to convert llm trafics from harnesses into trainable traces. And we use swebench\_harness to score the final edition patch in a fresh runtime. The main training hyperparameters are listed in Tab. 4.

<table><tr><td>Harness</td><td>Base</td><td>POLAR RL</td><td>Gain</td></tr><tr><td>Codex</td><td>3.8%</td><td>26.4%</td><td>22.6 ↑</td></tr><tr><td>Claude Code</td><td>29.8%</td><td>34.6%</td><td>4.8 ↑</td></tr><tr><td>Qwen Code</td><td>34.6%</td><td>35.2%</td><td>0.6 ↑</td></tr><tr><td>Pi</td><td>34.2%</td><td>40.4%</td><td>6.2 ↑</td></tr></table>

Table 1: SWE-Benc<sup>h</sup> Veri<sup>fi</sup>e<sup>d</sup> eva<sup>l</sup>uation. All rows start from the same Qwen3.5-4B base model and are trained over listed harnesses. Scores are pass@1 over the full benchmark, running on corresponding harnesses.

Fig. 6 shows the training reward for Codex, Claude Code, Qwen Code, and Pi. The Codex run begins near zero reward and rises steadily over training, with the last ten steps averaging 54.5% pass@1 reward compared with 9.5% over the first ten steps. Claude Code also improves substantially, rising from 28.8% over the first ten steps to 67.0% over the last ten steps. Qwen Code and Pi start from stronger native-harness priors: Qwen Code is noisier but rises from 61.6% to 66.0%, while Pi improves more clearly from 61.6% to 76.2% over the same first and last ten-step windows. These curves are consistent with the benchmark result in Tab. 1: Polar improves the same 4B base model under all four evaluated harnesses. The largest absolute gain appear in Codex, likely due to unfamiliar tool schemas

The evaluation isolates the value of harness-native RL. Under Codex, the 4B base model reaches only 3.8% pass@1 before training, but the Polar-trained checkpoint reaches 26.4%, a 22.6 point absolute gain. This large jump is expected: Codex presents an unfamiliar action protocol, context policy, and patch-submission style to a Qwen model that was not originally trained as a Codex-native policy. Polar keeps that harness unchanged and attaches the reward to the actual sampled tokens flowing through the Codex execution path, so GRPO optimizes the behavior the model must use at evaluation time. Under the native Qwen Code harness, the base model is already much stronger at 34.6%, and Polar still improves it to 35.2%, a 0.6 point gain. Claude Code also improves from 29.8% to 34.6%, adding 4.8 points, and Pi improves from 34.2% to 40.4%, adding 6.2 points. These results show that harness-native RL can deliver large adaptation gains for unfamiliar execution paths while still preserving gains when the base checkpoint is already well aligned with the harness.

## Trajectory builder ablation.

We ablate the trajectory builder under identical model, hardware, and topology settings, changing only whether captured completions are emitted as per\_request traces or merged by prefix\_merging. Fig. 5b shows a partial utilization profile. Over same three training steps, prefix\_merging reduces the trainer stream from 1,185 request-level updates to 218 merged-trace updates, cutting wall-clock time from 189.5 to 35.2 minutes (5.39×). prefix\_merging keeps rollout GPUs active with 87.7% average rollout utilization, compared with 20.4% average utilization for per\_request over the same period.

We also tried per\_request with outcome-reward broadcasting to every emitted trace, but observed significant reward hacking. The issue is noisy credit assignment: request-level traces can receive session-level credit without proper session normalization or an advanced process reward model. Those mechanisms are outside the scope of this work, but providing examples and tools for session normalization and PRM-style credit assignment is on our roadmap.

## 4.2. Ofline Data Generation

Beyond serving online RL rollouts, Polar can be repurposed as a distributed ofline data-generation service: a fixed checkpoint and harness are fanned out across the cluster, every session is journaled to disk, and the resulting traces are filtered and post-processed for downstream training. The same primitives that make Polar useful for RL—per-session container isolation, automatic retry, and gateway-mediated scheduling.

<table><tr><td>Repo</td><td>Attempts</td><td>Accepted</td><td>Rate</td></tr><tr><td>getmoto/moto</td><td>343</td><td>184</td><td>53.6%</td></tr><tr><td>python/mypy</td><td>257</td><td>101</td><td>39.3%</td></tr><tr><td>conan-io/conan</td><td>71</td><td>27</td><td>38.0%</td></tr><tr><td>pydantic/pydantic</td><td>81</td><td>24</td><td>29.6%</td></tr><tr><td>iterative/dvc</td><td>219</td><td>45</td><td>20.5%</td></tr><tr><td>pandas-dev/pandas</td><td>477</td><td>98</td><td>19.7%</td></tr><tr><td>dask/dask</td><td>141</td><td>25</td><td>17.7%</td></tr><tr><td>Total</td><td>1,638</td><td>504</td><td>30.8%</td></tr></table>

Table 2: Per-repository acceptance rates for SFT data generated by Polar with Qwen3.5-122B-A10B and the pi harness on SWE-Gym. “Accepted” means the agent’s patch passed both FAIL\_TO\_PASS and PASS\_TO\_PASS tests in the SWE-Bench evaluator.

## Case study: SWE-Gym SFT trajectories.

We used Polar to generate a supervised fine-tuning corpus of agentic software-engineering trajectories. The setup is intentionally minimal: a single 8×H100 SGLang serve job hosting Qwen3.5-122B-A10B (TP=8, max\_model\_len=32,768) drives the pi-coding-agent v0.67.68 harness against 1,638 instances drawn from seven SWE-Gym repositories. Each task runs in its own Apptainer SIF built from the SWE-Gym reference image with Node.js 22 and the harness layered on top, so the agent’s tool calls (bash, read, edit, write) execute against a fresh checkout of the target commit. Submission uses max\_concurrent=5–8, max\_retries=1, and a per-task timeout of 3,600 seconds; trajectories that finished with empty\_generation are retried once and the rest accepted as-is.

A trajectory is accepted into the SFT corpus if and only if the SWE-Bench evaluation harness reports the agent’s final patch as resolving every FAIL\_TO\_PASS test while leaving every PASS\_TO\_PASS test green. With this single-bit filter, Polar produced 504 accepte<sup>d</sup> trajectories <sup>f</sup>rom 1,638 attempts (30.8% acceptance), at a cost of roughly 64 GPU-hours on the interactive partition. Per-repository acceptance varies substantially with task dificulty (Table 2): bug-fix heavy repositories like getmoto/moto accept at over 50%, while data-frame and dataflow workloads with longer test suites accept below 20%.

## R<sub>e</sub>l<sub>ease</sub>d f<sub>orma</sub>t<sub>.</sub>

Each accepted row contains the SWE-Gym instance metadata (instance\_id, repo, problem\_statement, base\_commit, version) and the full multi-turn conversation as a list of OpenAI-style messages with role, content, tool\_calls, and tool\_call\_id fields, terminated by the assistant turn that produced the accepted patch. Trajectories are long: an average of 104 messages per session and 51 assistant turns, with a long tail above 200 turns. The corpus is released as a HuggingFace dataset under an Apache-2.0 license, with a 90/10 train/test split stratified by repository so that every repo is represented in both splits.<sup>3</sup>

We deliberately kept the filter narrow—a single binary verifier from the existing SWE-Bench harness—to keep the case study reproducible. The same Polar deployment can be re-used for richer ofline pipelines without changing the runtime: rejection sampling falls out of running multiple completions per prompt and keeping only those that pass the verifier; verifier-training data falls out of retaining the rejected trajectories alongside the accepted ones: preference data falls out of pairing accepted and reiected traces from the same prompt. Scaling the present run to the full 2,438-instance SWE-Gym set, swapping in stronger teachers, or adding additional harnesses (e.g. codex or claude\_code) requires no changes to the orchestration code—only additional submitter shards and the corresponding checkpoint.

## 5<sub>.</sub> Conclusion

Polar treats agent test-time environments as a first-class part of the RL system rather than an implementation detail to be ported into the trainer. Its central design choice is to move the integration boundary to the model endpoint: the harness runs normally, the proxy observes token-level model trafic, and the rollout service turns completed executions into trainable trajectories and rewards. This separation lets rollout scale independently from training and inference, while preserving the behavior of non-standard harnesses whose value often lies in their engineering details. We believe Polar opens a new paradigm for scaling agentic RL infrastructure in the modern era, and we are actively developing and maintaining the framework as the ecosystem evolves.

## Referen<sub>c</sub>e<sub>s</sub>

Agent Lightning Team. No more retokenization drift: Returning token ids via the openai compatible api matters in agent rl. https://blog.vllm.ai/2025/10/22/agent-lightning.html, 2025. URL https: //blog.vllm.ai/2025/10/22/agent-lightning.html. vLLM blog. 4

Anthropic. Claude code. https://www.anthropic.com/product/claude-code, 2026. Accessed: 2026-05-06. 2

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016. 2

Shiyi Cao, Dacheng Li, Fangzhou Zhao, Shuo Yuan, Sumanth R. Hegde, Connor Chen, Charlie Ruan, Tyler Griggs, Shu Liu, Eric Tang, et al. Skyrl-agent: Eficient rl training for multi-turn llm agent, 2025. URL https://arxiv.org/abs/2511.16108. 2, 3, 15

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023. 2

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2

Harbor Framework Team. Harbor: A framework for evaluating and optimizing agents and models in container environments. https://github.com/harbor-framework/harbor, 2026. URL https://github.com/har bor-framework/harbor. GitHub repository. 4

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.06770. 2, 4, 9

Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, and Yuqing Yang. Agent lightning: Train any ai agents with reinforcement learning, 2025. URL https://arxiv.org/abs/25 08.03680. 2, 4, 15

NVIDIA. Nemo gym: An open source library for scaling reinforcement learning environments for llm. https: //github.com/NVIDIA-NeMo/Gym, 2025. GitHub repository. 2

OpenAI. Codex: Ai coding partner from openai. https://openai.com/codex/, 2026. Accessed: 2026-05-06. 2

OpenClaw-RL Contributors. Openclaw-rl: Scalable rl in real-world agentic settings, 2026. URL https: //arxiv.org/abs/2603.10165. 15

Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with swe-gym, 2024. URL https://arxiv.org/abs/2412.21139. 2, 4

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkelev function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https://open

Prime Intellect. Prime-rl: Async rl training at scale. https://github.com/PrimeIntellect-ai/prime-rl, 2026. URL https://github.com/PrimeIntellect-ai/prime-rl. GitHub repository. 2, 3, 15

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2

Sijun Tan, Michael Luo, Colin Cai, Tarun Venkat, Kyle Montgomery, Aaron Hao, Tianhao Wu, Arnav Balyan, Manan Roongta, Chenguang Wang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. rllm: A framework for post-training language agents. https://github.com/rllm-org/rllm, 2025. URL https://github.com/r llm-org/rllm. GitHub repository. 2, 4, 15

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, et al. Opencua: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123, 2025. 2

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024. 2

Hao Zhang, Mingjie Liu, Shaokun Zhang, Songyang Han, Jian Hu, Zhenghui Jin, Yuchi Zhang, Shizhe Diao, Ximing Lu, Binfeng Xu, Zhiding Yu, Jan Kautz, and Yi Dong. Prorl agent: Rollout-as-a-service for rl training of multi-turn llm agents, 2026a. URL https://arxiv.org/abs/2603.18815. 2, 3, 15

Shaokun Zhang, Jieyu Zhang, Jiale Liu, Linxin Song, Chi Wang, Ranjay Krishna, and Qingyun Wu. Ofline training of language model agents with functions as learnable weights. In Forty-first International Conference on Machine Learning, 2024. 2

Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu, Zhiding Yu, and Guilin Liu. Nemotron-research-tool-n1: Exploring tool-using language models with reinforced reasoning. In The Fourteenth International Conference on Learning Representations, 2026b. URL https: //openreview.net/forum?id=yiE16lWzDj. 2

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Huang, Jef Sun, Chao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, et al. Sglang: Eficient execution of structured language model programs, 2024. URL https://arxiv.org/abs/2312.07104. 3

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023. 2

Zilin Zhu, Chengxing Xie, Xin Lv, and Contributors. slime: An llm post-training framework for rl scaling. https://github.com/THUDM/slime, 2025. URL https://github.com/THUDM/slime. GitHub repository. 3

## A. A<sub>pp</sub>endix

## A.1. Framework Com<sub>p</sub>arison

<table><tr><td>System</td><td>Async RL Support</td><td>Async Rollout Staging</td><td>Rollout as Service</td><td>Agent Harness Agnostic</td></tr><tr><td>POLAR</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>PRORL AGENT (Zhang et al., 2026a)</td><td>✓</td><td>✓</td><td>✓</td><td>✕</td></tr><tr><td>SkyRL-Agent (Cao et al., 2025)</td><td>✓</td><td>✓</td><td>✕</td><td>●</td></tr><tr><td>PRIME-RL (Prime Intellect, 2026)</td><td>✓</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>Agent Lightning (Luo et al., 2025)</td><td>●</td><td>✕</td><td>●</td><td>●</td></tr><tr><td>rLLM (Tan et al., 2025)</td><td>●</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>OpenClaw-RL (OpenClaw-RL Contributors, 2026)</td><td>✓</td><td>✕</td><td>✕</td><td>●</td></tr></table>

Table 3: Comparing ro<sup>ll</sup>out-system <sup>d</sup>esign c<sup>h</sup>oices. We find that modern rollout infrastructures should meet following criteria: async RL support means training can consume rollouts while generation continues under explicit policy-version or staleness handling; async rollout staging means rollout execution is decomposed into independently scheduled runtime-preparation, execution, post-run reconstruction/evaluation, and cleanup stages; rollout as service means a durable task API that is separable from a specific trainer loop; and nativeharness agnosticism means a CLI, SDK, or application harness can be trained without being reimplemented as the framework’s environment. ✓ denotes first-class support, ● denotes partial or planned support, and ✗ denotes that we did not find the property as a primary design contract in the referenced code or documentation.

The partial marks in Tab. 3 avoid treating adjacent mechanisms as absent. SkyRL exposes custom generators and Harbor integration, but native harnesses are not the default unit of rollout. Agent Lightning provides a rollout store, queue, runner control plane, and broad framework instrumentation, while its main boundary is trace/workflow observability rather than staged execution of opaque harness processes. rLLM includes fully asynchronous training and a model gateway that captures token IDs and log probabilities, but its rollout service abstraction is narrower than a distributed runtime lifecycle service. OpenClaw-RL decouples serving, rollout collection, judging, and training for real-world agent settings; its support is organized around OpenClaw and specific terminal, GUI, SWE, and tool-call recipes rather than arbitrary native harness submission.

## A.2. SWE-G<sub>y</sub>m GRPO H<sub>yp</sub>er<sub>p</sub>arameters

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Base checkpoint</td><td>Qwen/Qwen3.5-4B</td></tr><tr><td>Training data</td><td>NovaSky-AI/SkyRL-v0-293-data, train split, 293 tasks</td></tr><tr><td>Trainer</td><td>Slime asynchronous GRPO</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Rollout batch size</td><td>4</td></tr><tr><td>Samples per prompt</td><td>16</td></tr><tr><td>Trace construction</td><td>prefix_merging</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-6}$ </td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>TIS</td><td>Enabled</td></tr></table>

Table 4: Training <sup>h</sup>yperparameters <sup>f</sup>or t<sup>h</sup>e SWE-Gym GRPO experiments. The table reports ordinary policy-optimization and rollout parameters from examples/swegym\_slime\_grpo; cluster topology and worker placement are omitted.

## A.3. Re<sub>p</sub>resentative Task Pa<sub>y</sub>load

R<sub>epresen</sub>t<sub>a</sub>ti<sub>ve</sub> P<sub>o</sub>l<sub>ar</sub> T<sub>as</sub>k P<sub>ay</sub>l<sub>oa</sub>d

```json
{
    "task_id": "polar-swegym-{rollout_id}-{group_index}",
    "instruction": "Fix the issue in /polar/session/workspace.",
    "num_samples": 8,
    "timeout_seconds": 1200,
    "runtime": {
    "backend": "docker",
    "image": "{sample.metadata.runtime_image}",
    "network": "host",
    "workdir": "/polar/session/workspace",
    "prepare": [
    {
    "type": "exec",
    "command": "prepare repository, harness, and dependencies"
    }
    ]
},
"agent": {
    "harness": "codex",
    "model_name": "{served_model_name}"
},
"builder": {
    "strategy": "prefix_merging"
},
"evaluator": {
    "strategy": "swebench_harness",
    "refresh_runtime": true,
    "config": {
    "repo_dir": "/testbed",
    "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
    "instance": "{sample.metadata.instance}"
    }
},
"callback_url": "http://{trainer_host}:{callback_port}/callback/task_result",
"metadata": {
    "group_id": "{rollout_group_id}",
    "policy_version": "{policy_version}",
    "rollout_step": "{rollout_step}"
}
}
```

## A.4. Re<sub>p</sub>resentative Trace

## Re<sub>p</sub>resentative Trainer-Facin<sub>g</sub> Trace

```json
{
    "prompt_ids": [151644, 872, "..."],
    "response_ids": [9211, 374, 264, "..." , 151645, 271, 151644],
    "loss_mask": [1, 1, 1, "..." , 0, 0, 1],
    "response_logprobs": [
    {
    "token": "The",
    "token_id": 785,
    "logprob": -0.21
    ]
    ]
}
```

```json
},
{
    "token": "patch",
    "token_id": 10042,
    "logprob": -0.44
},
{
    "token_id": 151645,
    "logprob": 0.0
}
],
"prompt_messages": [
    {"role": "system", "content": "agent harness system prompt"},
    {"role": "user", "content": "task instruction and current tool state"}
],
"response_messages": [
    {"role": "assistant", "content": "sampled assistant turn"}
],
"tools": null,
"finish_reason": "stop",
"reward": 1.0,
"metadata": {
    "session_id": "session-123",
    "task_id": "polar-swegym-0001",
    "builder": "prefix_merging",
    "harness": "codex"
}
}
```

## A.5. Service API Summar<sub>y</sub>

The rollout service exposes a small asynchronous API:

• POST /rollout/task/submit: submit a non-blocking task request.

• GET /rollout/task/{task\_id}: poll task status, partial results, and final results.

• GET /rollout/status: inspect task states, node states, and pending sessions.

• POST /callbacks/session\_result: receive gateway session callbacks.

• POST /nodes/register and POST /nodes/{node\_id}/heartbeat: maintain gateway membership and scheduling metrics.

The gateway exposes a control surface for session creation, status, and deletion, plus a catch-all proxy surface for provider-style model requests. Session deletion is used by the rollout pipeline as best-efort cleanup after a terminal result has been persisted.