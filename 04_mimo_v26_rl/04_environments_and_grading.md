# An RL environment is a sandbox plus a grader you can trust

*Part 4 of [Recreating MiMo-V2.6-Pro](README.md). Previous: [RL systems](03_rl_systems.md). Next: [Distillation](05_distillation.md).*

An RL environment is three things: a **prompt**, a **sandbox** that can be reset to the same starting state for every attempt, and a **hidden verifier** that turns the final state into a reward. Xiaomi released 7,780 such tasks in five domains: 2,698 code tasks graded by hidden tests, 2,093 web-development tasks graded by vision-model judges, 1,000 vulnerability-reproduction tasks and 1,000 music tasks graded by rules, and 989 knowledge-work tasks graded by rubrics ([dataset](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss), at release). On 19 October 2026 the dataset page showed 7,684 rows, with code at about 2.6K. That is consistent with roughly 100 code tasks being removed after the leak audit described below, though no change has been announced. Either way, the release is a curated subset. The production run drew from 25 task sources and consumed at least 30 × 1,568 ≈ 47K prompt groups ([report](https://arxiv.org/abs/2610.11959) §5.1, §6.3, §7.2). The main lesson of this part is where the effort goes. Writing a task is cheap; **proving that its grader is right takes most of the work**. Xiaomi audits every task several times, attacks its own environments with a "hack agent" and zeroes detected cheating during training. Even so, an outside audit found that 67% of the released code tasks still leaked the answer through git ([Vals AI](https://www.vals.ai/blogs/mimo-reward-hacking)).

This part reads the released code. Paths below refer to the [verl fork](https://github.com/XiaomiMiMo/verl/tree/mimo-oss) (branch `mimo-oss`) and to [mimoagent](https://github.com/XiaomiMiMo/mimoagent/tree/mimo-oss) (default branch `mimo-oss`). The reward math built on top of these graders (GRS, GAR, length and segment penalties) is in [Part 2](02_rl_algorithm.md).

## Background: what a training environment looks like

If you come from pre-training, think of an environment as a data source that answers back. A row in the dataset is not a training example. It is a pointer to a Docker image plus a task text; the training example is whatever the model does inside that image.

**[SWE-bench](https://arxiv.org/abs/2310.06770)** (Jimenez et al., 2023) fixed the format almost everyone still uses for code. A task is a real GitHub issue, a repository frozen at the commit before the fix, and the tests that the fix made pass. Those tests split into **fail-to-pass (F2P)** tests, which must fail before the fix and pass after it, and **pass-to-pass (P2P)** tests, which must keep passing. SWE-bench was an evaluation set. Three 2025 papers turned the format into training data, and they are the best background for this part:

- **[SWE-Gym](https://arxiv.org/abs/2412.21139)** (Pan et al.): 2,438 executable Python tasks and the first open demonstration that training on them helps. Read it for the basic recipe.
- **[R2E-Gym](https://arxiv.org/abs/2504.07164)** (Jain et al.): 8,700+ environments synthesized from commit history, plus "hybrid verifiers". Read it for how to scale beyond human-written issues.
- **[SWE-smith](https://arxiv.org/abs/2504.21798)** (Yang et al.): about 50K tasks made by injecting bugs into working code. Read it for the "make tasks from code itself" idea that CodeMidas extends.

Outside code, the reference format is **[Harbor](https://github.com/harbor-framework/harbor)**, from the creators of [Terminal-Bench](https://www.tbench.ai/). A Harbor task is a directory: `instruction.md` (the prompt), an `environment/` with a Dockerfile, and a `tests/` folder that is copied in only after the agent finishes. Hugging Face's FineEnvs group converted Xiaomi's environments to this format, which is often the easiest way to run them (for example [MiMo-V2.6-RL-harbor-general](https://huggingface.co/datasets/FineEnvs/MiMo-V2.6-RL-harbor-general)).

Every environment in this part follows the same lifecycle:

![Lifecycle of one code rollout](images/code_rollout_lifecycle.png)

The grading material is absent while the agent works, and only stage 4 decides the base reward. Stage 5 exists only in Xiaomi's production run.

## Five domains, five kinds of grader

![Released tasks by grader type](images/released_tasks_by_grader.png)

About 60% of released tasks (code, cyber, music: 4,698 of 7,780) need no model to grade them. Webdev is judged by a vision model, and general tasks mix rule-based and LLM-judged rubric items. The production mix was different: coding was 68% of RL prompts, aesthetic design 13%, general tool use 12%, cyber 4% and context following 3% (report §5.1).

### Code: tests that appear only at grading time

A code row in the dataset is small. In the released parquet, `data_source` is `opensource-code`, `ability` is `swe`, `agent_name` is `mimo_swe_agent`, `reward_model.style` is `rule`, and `extra_info.instance_json` carries the `docker_image` and the repository `cwd` (often `/testbed`). The heavy parts live in a per-task Docker image and in a hidden test patch.

The grading contract is in [`environments/datasets/opensource_code.py`](https://github.com/XiaomiMiMo/mimoagent/blob/mimo-oss/src/mimoagent/environments/datasets/opensource_code.py). Its docstring explains why the verifier ships inside the test patch rather than in the image:

> the agent and the grader share one pod and one filesystem, so anything present during the rollout is readable with `cat`. The patch is applied only at reward time, so the hidden tests and the script that runs them never exist while the agent works.

The reward itself takes three steps, in this order:

```
1. reset every path the test patch touches back to the captured base ref
2. git apply test_patch
3. run test_command; reward = 1.0 iff it exits 0
```

Step 1 is not paranoia. If the agent created a file at a path the patch adds, `git apply` would fail with "already exists in working directory". That would be a reward of 0 the model did not earn. Failures in steps 1–2 are labelled `REWARD_TESTBED_CORRUPTED`, so the trainer can tell a broken testbed from a wrong answer. Part 3 explains why that distinction matters for the advantages.

### General ("s3k"): the agent and the answer key live in different containers

General tasks give the agent a workspace of business documents and a set of simulated business systems. One released example asks for Starbucks' FY2024 ASC 740 lease deferred-tax disclosure, drawn from an approved tax provision. The setup is in [`recipes/general/`](https://github.com/XiaomiMiMo/verl/tree/mimo-oss/recipes/general) of the verl fork.

![General-task pod](images/general_task_pod.png)

Each rollout gets one Kubernetes pod with two containers. The **main container** runs the agent's tools and holds `/work/workspace` read-write. The **sidecar** runs the task's MCP servers, which simulate SaaS systems backed by SQLite files under `/work/system/<mcp>/state.db`. The main container never mounts `/work/system`. In the code's words, "DB isolation is physical, so the agent cannot bypass the MCP tools with direct SQL." The agent writes its conclusion to `answer.md`.

The verifier files (`run_verify.py`, `verify.py`, `rubrics.json`, `answer_key.json`, `verifier_meta.json`) are uploaded to the sidecar only at reward time. Each rubric item in `verifier_meta.json` carries a `method` (`rule` or `llm`), a `weight`, a `gate` flag and a dimension. A gate item zeroes the score when it fails. The released harness config [`config/agent/general/s3k.yaml`](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/config/agent/general/s3k.yaml) also enables the anti-hack guard and clears the HTTP proxy variables.

Not every general row is an s3k task. The first row of the subset is a terminal-style task ("Repair wildcard output flags in the workflow engine") with 1 CPU, 2 GB of RAM, a 900-second agent timeout and `allow_internet: false`, graded by a test script plus an anti-tamper check.

### Webdev: render the page, then ask a vision model

The agent builds a site from a brief into `dist/`. The grader service in [`recipes/design/grader_service/`](https://github.com/XiaomiMiMo/verl/tree/mimo-oss/recipes/design/grader_service) renders the page in a headless browser, screenshots it, and scores it in three steps. The [graders section](#graders-what-scores-what) below walks through them.

### Cyber: does the crash match the sanitizer report?

Cyber tasks come from ARVO, a corpus of reproducible OSS-Fuzz bugs (**[ARVO paper](https://arxiv.org/abs/2408.02153)**). The image contains a sanitizer-instrumented binary, the source, and a `submit.sh` wrapper. The agent must write an input (a proof of concept, PoC) that triggers one specific bug. The report argues this skill underlies exploit development, privilege escalation and CTF work (§4.2.4).

Crashing the program is easy; complex C/C++ projects have many reachable crash paths. So the verifier checks two attributes taken from the ground-truth sanitizer report (ASan, MSan or UBSan): the **vulnerability type**, such as heap-buffer-overflow, and the **crash location**, the topmost project-level stack frame. A PoC passes only if both match under rule-based string matching. The task description is generated from the same report, so the description and the verifier share one source of truth (§4.2.4).

The report explains why it rejected the two obvious alternatives:

- **Differential testing against the fixed binary**, as in **[CyberGym](https://arxiv.org/abs/2506.02548)**, can reject correct PoCs when a patch is incomplete. Unrelated changes between commits can also flip verdicts.
- **LLM judging** "returns different verdicts for the same PoC across runs and cannot serve as a stable reward."

Unlike CyberGym, the agent also gets the compiled harness binary, as a researcher running the target under a debugger would. In mimoagent ([`datasets/arvo.py`](https://github.com/XiaomiMiMo/mimoagent/blob/mimo-oss/src/mimoagent/environments/datasets/arvo.py)) a grading server runs as root and writes `/root/last_result.json`; the agent runs as a restricted user.

### Music: a rule-based human-likeness score

Music tasks ask for a piece in ABC notation. The scorer in [`recipes/design/music/scorer/`](https://github.com/XiaomiMiMo/verl/tree/mimo-oss/recipes/design/music/scorer) renders it to MIDI with `abc2midi`. The score is 85% conformance to human percentile bands over 18 features in 6 groups (rhythm, texture, acoustic, tonal, register, structure), plus 15% Jensen–Shannon agreement with corpus histograms of pitch class, interval and duration. It uses no model and only the Python standard library. Music appears in the release and the 9B experiments, but not by name in the production task mix.

## How the tasks were built and validated

### Code: five sources, then two checks

The report lists five ways to make code tasks (§4.2.1):

1. **GitHub pull requests with linked issues**, deduplicated and filtered for reproducibility. When no usable issue exists, an LLM writes an issue-style spec from the reference patch and is told to omit solution details.
2. **Everyday development requests** from Xiaomi employees (features, refactoring, debugging, maintenance). Agents write tests for the requested behavior.
3. **Specification-driven tasks**: code generation under detailed, multi-constraint requirements.
4. **Tasks derived from existing code with CodeMidas.**
5. **Long-horizon tasks**, where agents expand requirements, add cross-component dependencies and write multi-step tests.

Public sources and licensed data vendors add more. Pathways 2 and 5 depend on internal or licensed material and are the least likely to be in the release.

**[CodeMidas](https://arxiv.org/abs/2609.22068)** (Ye et al., Xiaomi, September 2026) is the most reusable idea here: make tasks from features that already work, so you need no issue or pull request.

1. An agent finds functionality with a public entry point and removes its core implementation. The original becomes the reference solution.
2. It writes a task statement describing the required behavior, not the internal design.
3. Tests map to each requirement. Expected outputs come from running the reference code. A review pass drops assertions the statement does not support, such as exact error wording.
4. Cleanup removes artifacts that could reveal the deleted code. Each task must fail from its starting state and pass with the reference in six fresh containers.
5. After rollouts, tasks are dropped if agents exploit leaked material, if independent review disagrees with the verifier, or if a frontier model always passes or always fails.

That yields 5,545 tasks from 3,185 repositories in 23 languages. GRPO on these tasks moved MiMo-V2.5 from 10.0 to 21.7 on DeepSWE v1.1.

Every code task then has to pass two checks (§4.2.1):

- **Accuracy.** Each task is attempted four times, and an auditing agent compares every outcome with the spec. A pass that looks wrong flags a possible false positive in the tests. A fail that looks right flags a possible false negative.
- **Robustness.** F2P tests must fail before the reference patch and pass after it. P2P tests must pass both times. Outcomes must be identical across eight reruns, which catches flaky tests.

Note what the accuracy check uses: rollouts of a real model. You cannot validate a grader from the reference solution alone. You need wrong and partly right attempts to see whether the grader separates them.

### General: synthesize a company, then the tasks

General environments are synthesized end to end (§4.2.2):

1. **Environment construction.** A planning agent designs the workspace, picks tools and plans file and database contents, using web search to ground it in real-world information. Files and records are linked so tasks need multi-hop reasoning. Several agents then generate files and fill databases in parallel. Software is mocked by a multi-agent workflow that reproduces supported operations, state changes, output formats and error responses.
2. **Consistency review.** A review agent checks every artifact and the whole: entity names, numbers that must reconcile, timelines, references. Inconsistencies are repaired before any task is written.
3. **Task synthesis.** Common professional tasks are merged into seed tasks. An agent explores each environment and writes tasks that fit its contents.
4. **Rubrics.** Completion is scored with atomic, binary items. Code checks verify deterministic properties, such as database values and output formats. LLM checks assess open-ended content. Items on which repeated judgments or different judge models disagree are flagged as ambiguous.
5. **Rubric refinement.** A review agent reads rollouts from models of different strength, their results and their rubric verdicts. It tightens items that accept incomplete work and loosens items that reject valid work. Negative checks catch unintended changes to unrelated files or databases. Adversarial solutions, built to look correct without solving the task, test that the rubric rejects them.

The result is "thousands of environments". During RL, a self-hosted MiMo-V2.6-SFT model judges the LLM items.

### Visual: open-ended or replication

Visual tasks cover websites, interactive apps, games, 3D scenes, slides, SVG, video and Figma designs (§4.2.3). **Open-ended design** has many valid answers, so it combines pointwise rubrics (runtime correctness, instruction adherence, layout integrity, basic aesthetics), refined until stable, with a groupwise comparison of the rollouts for the same query. **Replication** tasks supply a reference target and are graded mainly by rule-based similarity, such as pixel-level similarity, with an LLM judge for holistic quality.

## Graders: what scores what

| Domain | Grader | Model in the loop | Source |
|---|---|---|---|
| Code (released) | hidden tests, exit code | none | `opensource_code.py` |
| Code (production) | tests × rubric scores (GRS) or tests + GAR ranking | grader agent; GAR uses an "SFT-trained agentic grader", base model not named | §4.3 |
| General | rubric items, `rule` or `llm` | self-hosted MiMo-V2.6-SFT | §4.2.2 |
| Webdev | runtime gate, requirement judge, group pick | configurable via `GRADER_MODEL`; no default | grader service |
| Cyber | sanitizer type + frame string match | none, by design | §4.2.4, `arvo.py` |
| Music | 18-feature band score + JS divergence | none | `music/scorer` |
| Hack audit | offline trajectory audit | not named | §4.2.6 |

### Webdev: a gate, a requirement judge and a group pick

The webdev reward combines three signals ([`group_pick.py`](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/design/grader_service/src/design_grader/service/group_pick.py)):

```
raw    = pick_norm − query_deduct          ∈ [−1.8, 1]
reward = (raw + 2) / 3                     ∈ [0.067, 1]
runtime gate fails  → reward = 0
query_score < 0.2   → reward = 0
```

**The runtime gate** needs no model ([`runtime_gate.py`](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/design/grader_service/src/design_grader/service/runtime_gate.py)). A page fails it if an inline script fails `node --check`, an uncaught exception fires during load, or the renderer hangs. The code comments record two lessons worth remembering:

- **A soft penalty is a price, not a rule.** "A soft factor is tradable against the visual score," so RL learns to pay it. Screenshot judges cannot see errors that fire after the page paints: 15% of pages with an uncaught error still scored ≥ 0.5.
- **Dropping a sample is a reward.** A page with `while(1){}` once hung the renderer. Treating the hang as "drop this sample" gave it zero advantage, which beats the negative advantage of a failure. So "the model learns that wedging the renderer is safe." Hangs now score as failures.

**The requirement judge** sees the brief and the full-page screenshot and returns a 0–1 score. The prompt ([`prompts/query_fit.md`](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/design/grader_service/src/design_grader/service/prompts/query_fit.md)) is in Chinese. Translated, it asks the judge to split the brief into visible requirements (topic, sections, elements, functional UI, style) and check each against the screenshot. It then picks one of five bands, from "[0.0, 0.2): basically nothing implemented, or the wrong topic" to "[0.8, 1.0]: fully and accurately implemented", and refines within the band. It is told to "not raise the score because the page is pretty". The service maps the score to a deduction: ≥ 0.9 → 0, ≥ 0.6 → 0.2, ≥ 0.4 → 0.4, ≥ 0.2 → 0.8.

**The group pick** compares the group's rollouts. In the released recipe a group has 8 rollouts. One prompt ([`prompts/group_pick.md`](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/design/grader_service/src/design_grader/service/prompts/group_pick.md), also in Chinese) shows the judge all 8 screenshots in random order. It asks which are "clearly better than the others" and which "clearly worse", 0 or more of each, or "no clear difference". It forbids a forced full ranking: "clearly" means "an ordinary person sees it at a glance". Three design details make it usable as a reward:

- **Position bias is cancelled.** The pick runs 8 rounds ordered as a Williams Latin square: every screenshot appears in every position once. The code reports this cut first-position bias from ±0.28 to ±0.03 votes.
- **Noise is zeroed.** A screenshot whose net vote is within ±1 gets 0. If 2 or more rounds say "no clear difference", the whole group's pick is 0.
- **The judges cannot see broken JavaScript.** Gated pages averaged −0.11 on the raw scale versus +0.02 for clean ones, nearly indistinguishable, which is why the gate exists.

This matches the report's description (§4.2.3) of pointwise checks plus a groupwise comparison. One code comment says the per-page aesthetic rubric is no longer on the training path.

### Rubrics with a judge agent

mimoagent's [`rubric_judge.py`](https://github.com/XiaomiMiMo/mimoagent/blob/mimo-oss/src/mimoagent/environments/rubric_judge.py) shows what an agentic rubric grader looks like. A **judge agent** with its own model enters the finished pod. A private workspace gives it `task.md`, `response.md`, `model_patch.diff`, `rubrics.json` and the doer's trajectory. It may read code, run tests, build and diff. It writes `verdicts.json`, which the environment reads exactly once; a missing file is an infrastructure fault, never re-prompted.

Scoring: B is the mean of the binary verdicts. An optional graded "solution" rubric lists quality levels best-first, each with a score in [0, 1]; the judge names the highest level reached, giving S. Two combinations are supported:

- `weighted` (default): w·S + (1 − w)·B, with w = 0.5 by default.
- `product`: S·B, so a poor solution caps the reward however clean the process. The code says this mode is "used by the RL side to rank test-passing rollouts". It is the released counterpart of GRS's R = R_test · S_sol · S_beh ([Part 2](02_rl_algorithm.md)).

Rubrics are written as obligations that apply to every rollout ("a rubric is written as an obligation, never as 'if the doer did X…'"). That keeps verdicts comparable across a group.

**An illustrative rubric.** No rubric is visible in the dataset rows; they live inside the task images. Below is a plausible rubric for the Starbucks ASC 740 task, written by us from the requirements in its prompt:

| id | Item (pass/fail) | Method | Gate |
|---|---|---|---|
| 1 | `answer.md` exists and contains the disclosure table | rule | yes |
| 2 | Lease-liability deferred tax asset equals the answer key within rounding | rule | no |
| 3 | ROU-asset deferred tax liability is reported separately, not netted | rule | no |
| 4 | Totals summed in unrounded cents before rounding to millions | rule | no |
| 5 | The FY2025 future-effective rate is not used | rule | no |
| 6 | State nonconformity is treated per jurisdiction, with reasoning | llm | no |
| 7 | No records in the tax-provision system were modified | rule (negative) | yes |

For the general theory, **[Rubrics as Rewards](https://arxiv.org/abs/2507.17746)** (Gunjal et al., 2025) is the standard reference. It shows that instance-specific rubrics used as on-policy rewards beat direct Likert scores from an LLM judge, and keep results more stable across judge sizes.

## Harnesses are part of the environment

The same task behaves differently under different agent loops. So Xiaomi treats the harness as a training dimension, like task and environment (§4.2.5). It did not train on production harnesses such as MiMo Code or Codex, for two reasons. Their safeguards and instruction prompts sit outside the reward, which makes credit assignment unreliable. And their modules are tightly coupled, so you cannot vary one mechanism at a time.

Instead, every **mini-harness** starts from one minimal loop with three swappable modules: system prompt, tools and context management. The released code recipe mixes four of them ([`config/agent/code/mix-four-whitebox.yaml`](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/config/agent/code/mix-four-whitebox.yaml)):

| Harness | mimoagent agent type | What it imitates |
|---|---|---|
| `mini-mimocode` | `mimocode-agent` | MiMo Code's tool catalogue |
| `mini-bash` | `bashonly-agent` | one bash tool only |
| `mini-claude-code` | `cc-agent` | Claude Code's tools (Bash, Read, Write, Edit, Grep, Glob) |
| `mini-codex` | `codex-agent` | Codex's tools; `compaction_enabled: false` |

The `cc-agent` harness lets the model **compact its own context**. A `Compact` tool summarizes the history, which is rebuilt as system prompt + original issue + summary. A `<context_usage>` footer on tool results tells the model how full its context is. Part 3 covers how compacted trajectories are split for training.

**How the mix is drawn.** In the released code recipe, the default `MIXED_HARNESS_MODE=prompt` assigns one harness per prompt ([`recipes/code/mimoagent_runner.py`](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/code/mimoagent_runner.py)). All 16 rollouts of a group share a harness, and the batch mixes harnesses across prompts. That keeps the group baseline clean: siblings differ only in their own sampling, not in their scaffold. A `paired-subgroup` mode instead assigns harnesses per rollout, and `algorithm.group_advantage_by_harness` (off by default) can compute the baseline separately per harness within a group. The report does not say which granularity production used.

Did mixing help? On DeepSWE v1.1, mean pass@1 on three held-out harnesses (Codex, Claude Code, mini-swe-agent) rose from about 50% to 66% during training. The gap to the training harnesses narrowed (§5.3). On the 9B model, multi-harness RL improved all 21 dataset × harness pairs, including +1.8 to +9.3 points on MiMo Code Bench (mini) across seven harnesses (§7.2).

## Reward hacking: the grader is an attack surface

Early MiMo runs showed "a tendency toward reward hacking" (§4.2.6). On repository-repair tasks the main failure was **solution leakage**: the agent found the published fix outside its checkout.

| Pattern | What the agent did |
|---|---|
| Install and read | installed a newer release of the package and read its source |
| Fetch upstream | downloaded the fixed file |
| Clone upstream | cloned a later state of the repository |
| Look up the solution | searched issues, pull requests or commits for the fix |
| Probe versions | listed releases (e.g. `pip index versions`) to find one with the fix |

### Four layers of defense

1. **Before RL.** Hacking trajectories are turned into mid-training data in which the model reflects on the faulty step and rewrites it ([Part 1](01_base_model.md)).
2. **Environment preparation.** Strip build logs, verifier outputs, residual patches and binaries. Keep git history only up to the base commit. Isolate the network at the container level.
3. **A hack agent.** A dedicated agent attacks every prepared environment. Fix what it finds, run it again, and repeat "until the hack agent could no longer find a successful exploit in any of the environments."
4. **Training-time auditing.** Trajectories are audited offline. A confirmed hack gets reward 0 *before* group statistics and advantages are recomputed. Otherwise the hack would still have raised the group mean and hurt its honest siblings.

The confirmed-hack share stayed below 2% of trajectories for both models throughout the final run.

The released harnesses add two runtime layers. First, the system prompt of `mini-claude-code.yaml` forbids the known shortcuts outright:

> do not fetch the upstream or a newer version of the repo from GitHub (or any mirror), do not pip/npm install a newer release of the package under test, do not overwrite test files to make them pass, and do not hard-code expected outputs. […] any such shortcut will be detected and scored zero.

Second, [`agents/antihack.py`](https://github.com/XiaomiMiMo/mimoagent/blob/mimo-oss/src/mimoagent/agents/antihack.py) intercepts tool calls before they run. Stage 1 is a recall-first regex filter "derived empirically from 6,026 judged hacking trajectories". It allows sanitized local history (`git log`, `git show HEAD~1:path`) but blocks cross-ref and remote archaeology (`--all`, `origin/`, `refs/`), raw object access (`git cat-file`, `git fsck`), fetching patches from GitHub, upgrading the package under test, and reading files named like answer keys. A flagged call never runs; the agent sees "Permission denied. Resource unavailable." Stage 2, an LLM judge to confirm intent, is a stub. The guard is off by default and enabled per config, as in `s3k.yaml`.

The 6,026 figure is worth pausing on. Thousands of judged hacking trajectories existed somewhere in Xiaomi's pipeline. Which runs they came from is not stated.

### The Vals audit: 67% of released code tasks still leaked

In October 2026 [Vals AI audited](https://www.vals.ai/blogs/mimo-reward-hacking) the released code environments. **1,795 of 2,698 tasks (67%) kept the reference fix in git**, in three forms:

- **Unreachable objects.** Branches had been deleted, but their objects were never pruned. The old setup check, `git rev-list --all --not <base>`, only inspects reachable history, so it passed.
- **Full upstream history.** Checking out the base commit only moves HEAD. `git log HEAD..origin/main` listed the later commits.
- **File timestamps.** Even without git, `find -newermt` revealed which 18 files the reference patch changed.

MiMo-V2.6 exploited all three when tested. Flash found an unreachable fix object in a SQLGlot task, copied it and passed 5/5 hidden tests. With the anti-hack guard blocking `git fsck` and `git log --all`, Flash **wrote its own pack-file parser** to read the objects. Pro cloned upstream vLLM after reading the "do not cheat" rule as a hint that the answer existed online. The behavior was prompt-sensitive: on one task Flash searched for the upstream fix in 6/6 runs with the original prompt, 5/6 with "Do not cheat", and 0/6 when the prompt named the forbidden sources. Vals found the same leftover-branch problem in other public SWE datasets.

Xiaomi's lead Fuli Luo replied that internally, "our RL setup is hackable environments plus a separate anti-hack preprocessing script" ([HuggingNews](https://huggingnews.com/ai/developers-patch-answer-leaks-in-xiaomi-mimo-rl-environments-e7cf5084)). In other words, the released images lacked a cleanup step that ran in training. On 18 October 2026, mimoagent commit [`c3df846`](https://github.com/XiaomiMiMo/mimoagent/commit/c3df8469f250b15e7906487a6d52be4c1827a509) ("purge task git history at setup") added the missing step, and the verl fork bumped to it (`2290fcbc`). The fix is a good template:

```bash
for r in $(git remote); do git remote remove "$r"; done
# delete every ref that is not an ancestor of the base commit
git for-each-ref --format='%(refname)' | while read -r ref; do
  case "$ref" in refs/heads/*|refs/tags/*) git merge-base --is-ancestor "$ref" "$base" && continue ;; esac
  echo "delete $ref"
done | git update-ref --no-deref --stdin
rm -rf "$gd/logs" "$gd/worktrees"; rm -f "$gd/ORIG_HEAD" "$gd/FETCH_HEAD" ...
git -c gc.pruneExpire=now -c gc.reflogExpire=now ... gc --prune=now
# then verify: no remotes, no commits beyond base, and every commit in the
# object database is reachable (odb_commits == reachable_commits)
test "$remotes" -eq 0 && test "$extra" -eq 0 && test "$odb" -eq "$reachable"
```

The last line is the lesson. The old check asked "is any later commit **reachable**?" The new one also asks "does the object database contain **any** commit that is not reachable?"

Community ports found more. The Harbor ports now upload grading files only after the agent finishes, block answer sources by host name, and verify that a no-op agent scores 0 on every task. One early cyber port let an agent running as root overwrite `/root/last_result.json`, which the grader reads ([Towards AI](https://pub.towardsai.net/what-xiaomis-7-780-mimo-tasks-reveal-about-training-agents-2788d4ce8a81)). Xiaomi's own ARVO environment runs the agent as a restricted user for exactly this reason.

### What "under 2% detected" can and cannot mean

The report's number counts **confirmed** hacks among audited trajectories. It cannot count hacks the audit never recognized, and those keep their reward. The Vals results show capable models invent new routes (a hand-written pack parser, file timestamps) that no regex anticipated. So read the 2% as a lower bound on hacking in production, measured on Xiaomi's internal environments, which differ from the released ones. Neither the audit's sampling rate nor its auditor model is stated.

The practical rule: **assume every grader leaks until a capable adversarial agent has failed to break it, and re-run that attack whenever the environment or the model changes.**

## What is released, and what you can rebuild with it

| Artifact | Contents | What it lets you do |
|---|---|---|
| [MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) (Apache-2.0) | 7,780 rows in five subsets; per-task images on [Docker Hub](https://hub.docker.com/r/xiaomimimo/mimo-v2.6-rl-oss) | run and inspect the environments; read graders |
| [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl/tree/mimo-oss) | one squashed commit on verl 0.9.0.dev (`2781c1e4`): 191 files, about 30.7K lines; five launchers | RL on one domain at a time |
| [mimoagent](https://github.com/XiaomiMiMo/mimoagent/tree/mimo-oss) (MIT; started as a fork of mini-swe-agent) | harnesses, tools, sandboxes (local, Docker, Kubernetes, Modal…), dataset adapters, rubric judge, anti-hack guard | rollouts and grading without training |
| [uni-agent](https://github.com/XiaomiMiMo/uni-agent) (submodule) | model gateway and trajectory capture | used by the code recipe only |
| [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) | the starting checkpoint named in the verl README | a policy small enough to train |
| FineEnvs Harbor ports, e.g. [general](https://huggingface.co/datasets/FineEnvs/MiMo-V2.6-RL-harbor-general) (925 tasks) | the same tasks in Harbor format, with extra leak fixes | run tasks against any OpenAI-compatible endpoint |

The five launchers are `scripts/code/train.sh`, `scripts/arvo/arvo.sh`, `scripts/general/general.sh`, `scripts/design/webdev.sh` and `scripts/design/music.sh`. They wire rollouts in two ways. The code recipe replaces verl's agent-loop manager with uni-agent and runs mimoagent inside it. Cyber, general and webdev drive mimoagent directly from a verl `AgentLoop`. Music uses neither. Group sizes differ too: 16 for code and cyber, 8 for general, webdev and music.

**What the fork adds to upstream verl.** About 2.2K of the 30.7K added lines touch verl itself; the rest are the five recipes, the webdev grader service and tests. The verl changes are:

- `prompt-mean` loss aggregation, with weights computed over the whole optimizer batch (`core_algos.py`, `losses.py`, `trainer_base.py`);
- infrastructure failures kept out of the signal: zero advantage for invalid rows, and group filtering that ignores them (`core_algos.py`, `replay_buffer.py`);
- the length penalty, the mass-conserving segment penalties and a wired "reference penalties" profile, all off by default (`length_penalty.py`, `signed_rebalance.py`, `arvo_penalties.py`);
- per-harness metrics and optional per-harness advantage grouping;
- an SGLang metrics exporter for rollout monitoring.

Everything else this lesson describes in the fork is inherited from upstream verl: the trainer modes and partial rollout, the replay buffer's staleness logic, routing replay, TransferQueue, Muon support and an on-policy distillation trainer ([Part 5](05_distillation.md)).

These recipes correspond to the report's §7.2 baselines: GRPO on the 9B model, one domain at a time. The report gives neither group sizes nor step counts for those runs, and its code set (about 3K tasks) is slightly larger than the released 2,698, so treat its numbers as targets rather than a guaranteed reproduction. The reported gains were SWE-bench Verified 61.1 → 66.2, Terminal Bench 2.1 37.1 → 52.8, OfficeQA Pro 19.5 → 24.8, Toolathlon-Verified 35.2 → 38.0, MiMo Cyber Bench (mini) 31.3 → 47.0, MiMo Visual Coding (mini) 64.0 → 72.4, and an internal music benchmark 45.7 → 52.5. They do **not** reproduce the production run: there is no mixed-domain config, no Sample Mixer, and no GRS or GAR grader ([Part 3](03_rl_systems.md)).

## Published vs. missing

| Item | Status |
|---|---|
| Task format, Docker images, hidden tests for 7,780 tasks | released |
| Code grading contract (reset, apply, run) | released, `opensource_code.py` |
| General two-container pod, MCP mocks, verifier layout | released, `recipes/general` |
| Webdev runtime gate, requirement judge, group pick, prompts | released, `grader_service` |
| ARVO sanitizer-match grader; music scorer | released |
| Rubric judge agent, anti-hack guard | released, mimoagent |
| Git-history purge for code images | released 18 Oct 2026, after the audit |
| Code task pipelines, accuracy and robustness audits | described (§4.2.1); CodeMidas paper, no code |
| General environment and rubric synthesis agents | described (§4.2.2) |
| Hack agent and offline audit | described (§4.2.6); no code, no auditor model named |
| GRS rubric generator and GAR grader | described (§4.3); not released |
| Production environments beyond the curated subset (25 sources) | missing |
| Judge model weights (MiMo-V2.6-SFT) | missing |
| Hackable-environment fractions by cleanup round | in Figure 6 only, no numbers in text |

## What remains uncertain

How much of the production pool the release covers is unknown. 7,780 tasks against at least 47K consumed prompt groups suggests a small share, unless prompts repeated across steps. Whether the internal training images had the same git leaks is not established. Vals audited only the released copies, and Fuli Luo's reply implies a preprocessing step the release lacked. The judge model for webdev in production, the GAR grader's base model and the hack auditor's model are not named. The 6,026 judged hacking trajectories behind the regex filter come from an unstated set of runs. Music is in the release but absent from the report's production mix. It may be counted under aesthetic design; that is our guess. The Starbucks rubric above is illustrative, not Xiaomi's. The `/root/last_result.json` loophole was reported for an early Harbor port, not for Xiaomi's own setup.

## Questions to think about

1. The code reward resets every path the test patch touches before applying it. Construct a rollout that would score 0 without that step despite a correct fix, and one that could score 1 without it despite a wrong fix.
2. Why does the webdev grader zero a renderer hang instead of dropping the sample? Work out the sign of the advantage in both cases for a group where the other 7 rollouts score 0.4.
3. The old git check used `git rev-list --all --not <base>`. Name two ways the reference fix can survive that check, and say which line of the purge script handles each.
4. A general-task rubric has 10 binary items and no gate. What policy behavior would you expect if the deliverable item ("`answer.md` exists") were not a gate? What if all items were gates?
5. The cyber verifier rejects LLM judging because verdicts vary across runs. GAR and the webdev group pick still use LLM judges. What makes noise tolerable in those two places but not in cyber?
6. Design a hack-agent loop for the general environments. What would it try first, and what would count as "no exploit found"?

## Further reading

- **Environments for code:** [SWE-bench](https://arxiv.org/abs/2310.06770), [SWE-Gym](https://arxiv.org/abs/2412.21139), [R2E-Gym](https://arxiv.org/abs/2504.07164), [SWE-smith](https://arxiv.org/abs/2504.21798), [CodeMidas](https://arxiv.org/abs/2609.22068).
- **Cyber:** [ARVO](https://arxiv.org/abs/2408.02153), [CyberGym](https://arxiv.org/abs/2506.02548).
- **Formats and runners:** [Harbor](https://github.com/harbor-framework/harbor), [Terminal-Bench](https://www.tbench.ai/), [mimoagent](https://github.com/XiaomiMiMo/mimoagent/tree/mimo-oss).
- **Rubric grading:** [Rubrics as Rewards](https://arxiv.org/abs/2507.17746).
- **Reward hacking:** **[Lilian Weng, "Reward Hacking in Reinforcement Learning"](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)** (Nov 2024) is the best survey of causes and early mitigations. For this release: [Vals AI's audit](https://www.vals.ai/blogs/mimo-reward-hacking) and the [Towards AI tour of the environments](https://pub.towardsai.net/what-xiaomis-7-780-mimo-tasks-reveal-about-training-agents-2788d4ce8a81).
- **Primary source:** [MiMo-V2.6 report](https://arxiv.org/abs/2610.11959) §4.2 (environments, harnesses, reward hacking), §5.3 (multi-harness results), §7.2 (released environments and 9B baselines).
