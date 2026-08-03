# IdeaAmbig (ARR) — Chapter Plan + INSIGHT Collection

生成时间:2026-07-23 | 截止日期:本周内(2026-07 底前)

---

## 0. 全篇论证主线(Throughline)— 已确认

**主框架:三级能力递进衰减(Recognize → Localize → Resolve)**
模型在"识别规格是否就绪"上有一定能力,但定位具体缺陷位置的能力骤降,而提出真正能解决缺陷的澄清动作的能力进一步骤降。三张表构成一条递减曲线:

- Table 2(Task 1 Readiness):Label 层面尚可,但 PairAcc 普遍很低(个位数~20%出头)
- Table 3(Task 2 Localization):Label-L1 Acc ~50-60%,但 Loc-Acc(真正对准目标缺陷)骤降至 ~3-11%
- Table 4(Task 3 Resolution):待补全,预期作为三级递进的最后一环收尾

**支撑证据(非主框架,是 Tier 1 内部的一个次级论点):**
模型的失败不是随机的,而是系统性地"过度自信/过度接受"(over-accept)——UnsafePass 普遍偏高(如 Qwen3-8B 94.48%),说明模型倾向于把不完整规格当作完整规格来处理,而不是主动怀疑。

> **写作提示**:Intro 的最后一段("Our main contributions")、§6 Results 的开篇总起句、§7 Conclusion 的第一句,三处应该用同一套语言呼应这个三级递进框架,读者读完摘要就应该能预判到结果的形状。

---

## 1. Abstract — 需要小修 + 一个待决策项

**现状**:整体框架没问题,但有 `[TODO: rewrite later]` 挂在 meta-evaluation of LLM-as-Judge 那句结论上。

**⚠️ 风险项(见 INSIGHT #3)**:这句话目前**没有数据支撑**,而且你说 Meta-Evaluation 这部分"需要重新设计"。摘要里的这句话在结果定稿前不能维持原状。

**行动项**:
- [ ] 等 §5/§6 的 Meta-Evaluation 部分定稿(或决定砍掉)后,回来同步改这句摘要
- [ ] 摘要其余部分与三级递进框架的措辞对齐(可以在"generate fluent but plausible clarifications... fail to recover the target missing detail faithfully"这句后面,更明确点出"identification-localization-resolution"的递进,而不只是笼统提"substantial challenges remain"）

---

## 2. Introduction — 基本完成,两处小尾巴

**现状**:论证逻辑完整(SE 文献 → 研究点子的特殊性 → IdeaAmbig 定位 → 三个贡献点),不需要大改。

**行动项**:
- [ ] 处理 `[TODO: may have more sources]`(第2页,"Scientific-discovery benchmarks assess whether LLMs can retrieve inspirations..."这段前)——确认是否需要再补引用,还是删掉这个 TODO 标记
- [ ] 核对"Our main contributions"三个 bullet 是否已经暗示了三级递进(目前看是按 Task1/Task2/Task3 顺序写的,天然对齐,不需要改动逻辑,只需微调措辞让递进感更强)

---

## 3. Related Work — 需要新增一段(Manasi 批注,已确认要做)

**现状**:三篇 SE 代码模糊性文献(Larbi 2025, Yang 2026a, Vijayvargiya 2025)目前只在 §1 Intro 里被笼统引用了一次,§2 Related Work 的 "Ambiguity, underspecification, and clarification" 段落只引了 zhang2025modeling 和 wang2025learning——**这两组文献不是同一批**,Manasi 要的区分段落还不存在。

**待写内容(区分逻辑建议)**:
> Code-ambiguity 工作(Larbi et al., Yang et al., Vijayvargiya et al.)研究的是**程序任务规格**层面的模糊性——即给定一个相对具体的编程任务描述,其中的模糊/矛盾/不完整如何影响代码生成正确性。IdeaAmbig 研究的是更上游的**研究想法/方法规格**层面的模糊性——在代码或算法尚未被要求实现之前,一个高层研究想法本身是否包含足够的方法级细节(算法步骤、训练/推理逻辑、评估协议)供忠实实现。换句话说,code-ambiguity 文献关心"给定任务,模糊性如何影响写代码";IdeaAmbig 关心"任务本身要不要被写代码之前先弄清楚"——是 idea-to-implementation handoff 中更靠前的一步。

**行动项**:
- [ ] 在 §2 "Ambiguity, underspecification, and clarification" 段落中插入 2-3 句,引用 Larbi/Yang/Vijayvargiya 三篇,并加上述区分逻辑
- [ ] 检查插入后是否与 §1 Intro 里对这三篇的引用重复/冲突,若重复,考虑把 Intro 里的引用简化,把详细区分放在 Related Work(职责分离:Intro 提出问题背景,Related Work 做立场区分)

---

## 4. §3 IdeaAmbig(Taxonomy + 数据构建)— 已完成,无需改动

内容扎实(taxonomy 定义、数据来源、构建流程、Figure 1/Table 1 都齐全)。**不建议本周动它**,时间应优先给 §5/§6。

---

## 5. §4 Task Formulation — 已完成,无需改动

三个任务定义、指标公式(GBRA, PairAcc, MacroDRR, Macro-CAS)都写好了。**不建议本周动它**。

---

## 6. §5 Experimental Settings — 全部子标题都是空的,需要从零写

当前子标题(§5.1 Evaluated Models / §5.2 Evaluation / Human Evaluation / Meta-Evaluation / LLM-as-Judge)都还是空壳。根据你提供的素材,逐个来看:

| 子节 | 素材状态 | 本周行动 |
|---|---|---|
| **Evaluated Models** | ✅ 现成——就是 Table 2/3/4 里的 8 个 baseline(GPT-5.6-Sol, Claude Sonnet 5, Gemini 3.1 Pro Preview, DeepSeek V4 Pro; Qwen3-8B, Qwen3-32B, DeepSeek-R1, GPT-OSS-120B) | 直接写:模型版本号、访问方式(API/本地权重)、解码设置(温度、max tokens)、是否用了固定 prompt 模板 |
| **Evaluation**(评测协议) | 部分现成——zero-shot、direct-resolution vs interactive clarification 两种设置在摘要里已提到 | 写清楚两种设置的具体操作定义;呼应 Task 1/2/3 各自的评测流程(§4 已经定义了指标,这里只需要说"如何跑" |
| **Human Evaluation**(质量控制) | 🟡 本周能完成 | 这周做完后,按标准写法:抽样规模、标注人数、与模型一致性(agreement)、Cohen's kappa(参考你附录 Table 5 taxonomy construction 里已经用过这套写法,直接复用格式) |
| **Meta-Evaluation** | 🔴 需要重新设计,详见 INSIGHT #3 | 本周需要先做决策:①重新设计后有数据 → 补一个完整小节;②来不及 → 从这周的写作范围里砍掉,摘要同步降级措辞;③保留但转为"initial/exploratory"表述,弱化结论强度 |
| **LLM-as-Judge** | ✅ 有现成协议——附录 C.4 已经写了 Task3 的多维度 judge 协议(target relevance / sufficiency / feasibility / no-assumption / atomicity) | 这里只需要把附录 C.4 的协议**搬一个精简版到正文 §5**(正文讲"用了什么协议",附录保留完整 prompt),不需要新写内容 |

**优先级建议(本周内)**:Evaluated Models(最快)→ LLM-as-Judge(复用附录,快)→ Evaluation 协议(中等)→ Human Evaluation(等结果出来再写,预计本周末)→ Meta-Evaluation(先做决策,见下方 INSIGHT #3 的三选一)

---

## 7. §6 Results — 完全空白,需要从零写,且依赖 Table 4 补全

**写作结构建议(呼应三级递进主线)**:

1. **开篇总起句**:直接点出三级递进的核心发现(不要用"In this section we present results"这种 throat-clearing 开头,直接给结论)
2. **Task 1 段落**:解读 Table 2——PairAcc 低、UnsafePass 高,说明"过度接受"是主要失败模式;点名表现最好/最差的模型对比(如 Qwen3-32B 34.36 vs Qwen3-8B 5.52,说明模型规模/训练方式的影响)
3. **Task 2 段落**:解读 Table 3——Label-L1 尚可但 Loc-Acc 骤降,论证"知道类型≠找得到位置"
4. **Task 3 段落**:解读 Table 4(等补全后)——论证"即使定位到,也未必能提出真正能解决问题的澄清动作";可以用附录 Figure 3 的成功案例做对照,展示"好的澄清动作长什么样",反衬平均水平的不足
5. **跨任务小结**:一句话总结三级衰减曲线,呼应 abstract 和 conclusion

**依赖**:Table 4 其余 7 个模型 + Synth. 列数据(预计本周补齐)。

**行动项**:
- [ ] 等 Table 4 补全后再动笔(不要用占位数字先写,避免叙事和数字对不上返工)
- [ ] 补充跨模型的对比分析(哪类模型在哪个 tier 掉得最快,是否 open-weight vs proprietary 有系统差异——从 Table 2/3 数据看,Gemini 3.1 Pro Preview 在 GBRA/Macro-F1 上相对最强,这可以作为一个值得展开的观察点)

---

## 8. §7 Conclusion — 完全空白,按你选定的方向写

**确认方向**:聚焦在 benchmark 本身的贡献 + future work(taxonomy 可扩展到其他学科)。

**建议结构**(2-3 段,不需要长):
1. 重申 IdeaAmbig 是什么、解决了什么空白(idea-to-implementation handoff 的规格诊断)
2. 一句话重申三级递进的核心发现,呼应 Results
3. Future work:①taxonomy 目前只覆盖 AI/NLP/ML(Limitations 里已提到),可扩展到其他科学领域;②可以探索用 IdeaAmbig 的诊断能力去改进真实的 research-agent pipeline(呼应 Intro 里"reliable research automation"的动机,但不要过度引申——你选的是保守方向,所以这里点到为止即可,不要写成"我们相信这将彻底改变…"这种夸大表述)

---

## 9. Limitations — 已写一半,需要补完

**现状**:已有 4 点(领域局限于 AI/NLP/ML;clarification 非唯一性;真实 gap 收集难;不做端到端执行验证)。

**行动项**:
- [ ] 如果 Meta-Evaluation 本周被砍掉或弱化(见 INSIGHT #3),在 Limitations 里加一句说明("we did not conduct a full meta-evaluation of automated judges against expert judgments; this is left to future work" 或类似措辞)
- [ ] 如果 Human Evaluation 只能做部分抽样,也在这里注明抽样规模的局限性

---

## INSIGHT Collection(本次对话中浮现的关键洞察,供后续阶段引用)

1. **主框架已确认**:三级递进(识别→定位→解决,能力逐级衰减)为主线,过度自信(over-accept)是 Tier 1 内部的支撑证据而非并列主线。所有章节的措辞应向这条主线对齐。

2. **Manasi 批注未真正解决**:三篇 SE 代码模糊性文献目前只在 Intro 里笼统带过,Related Work 里要求的显式区分段落还不存在。这是审稿人明确会检查的点,必须本周补上(§3 已给出具体行文建议)。

3. **🔴 Meta-Evaluation 是本次规划中最大的风险项**:Abstract 里已经写了"automated evaluators only partially agree with expert judgments"这个结论性论断,但对应的 Meta-Evaluation 设计目前"需要重新设计",没有确定数据支撑。这是一个**必须在本周决策**的分支点,建议在动笔写 §5/§6 之前,先决定走哪条路(保留完整/弱化/砍掉),否则 Abstract、§5、Limitations 三处都会被这一个决策连带影响,存在返工风险。

4. **Table 4 完成度是 §6 写作的硬性阻塞项**:目前只有 DeepSeek V4 Pro 一行(Real 列),其余 7 个模型 + Synth. 列数据预计本周内跑出。§6 Results 的 Task 3 段落必须等这批数据到位后再写,避免先写叙事后数字对不上。

5. **Human Evaluation 预计本周能完整完成**,可以按标准格式写(抽样规模、agreement、Cohen's kappa),不需要降级处理。

6. **LLM-as-Judge 协议是"搬运"而非"新写"**:附录 C.4 已经有完整的多维度 judge 协议描述,§5 正文只需要精简版,不产生新内容负担,可以优先快速完成,腾出时间给风险更高的 Meta-Evaluation 决策和 Table 4 等待。

---

## 本周执行优先级建议(依赖关系排序)

1. **今天/明天**:决策 Meta-Evaluation 走向(风险项 #3)→ 决定后立刻同步修改 Abstract 的 TODO 句子
2. Related Work 补 Manasi 差异化段落(不依赖任何外部数据,可以立刻做)
3. §5 Evaluated Models + LLM-as-Judge(现成素材,快速完成)
4. Human Evaluation 收尾 → 写入 §5
5. 等 Table 4 补齐 → 写 §6 Results(Task 1/2 段落可以先写,Task 3 段落等数据)
6. §7 Conclusion(依赖 §6 结论定稿后再写,保持呼应一致)
7. Limitations 收尾(依赖 Meta-Eval/Human-Eval 最终状态)
8. 全篇通读:检查 Abstract → Intro → Results → Conclusion 四处的三级递进措辞是否前后呼应
