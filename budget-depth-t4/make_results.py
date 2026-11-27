"""Create the compact Chinese readout from independently verified aggregates."""
import csv
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
def rows(name):
    with (HERE/name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    rows_a, rows_g = rows("admission.csv"), rows("depth-length.csv")
    guards = {(r["domain"],int(r["budget"]),r["guard"]):r for r in rows_a}
    grid = {(r["domain"],int(r["budget"]),int(r["candidates_per_retriever"]),int(r["encoder_tokens"]),
             int(r["rerank_depth"]),int(r["reranker_tokens"])):r for r in rows_g}
    domain_names = {"scifact":"SciFact","nfcorpus":"NFCorpus","fiqa":"FiQA"}
    # The release uses domain-specific millisecond budgets. Load each from the observed groups.
    bs = {d: sorted({b for domain,b,_ in guards if domain==d}) for d in domain_names}
    recorded_run = hashlib.sha256((HERE/"observations.zip").read_bytes()).hexdigest() == "5fc07d3269dc118b88ee1152a268471b21c6b385b5328dd5b95b3edc1f317327"
    body = []
    line = body.append
    def table(columns, data):
        line("|"+"|".join(columns)+"|")
        line("|"+"|".join(["---"]*len(columns))+"|")
        for r in data:
            line("|"+"|".join(map(str,r))+"|")
        line("")
    def metrics(r):
        n = int(r["queries"])
        return [f'{float(r["ndcg"]):.4f}', f'{float(r["utility"]):.4f}', f'{float(r["mean_ms"]):.2f}',
                f'{float(r["p95_ms"]):.2f}',f'{r["overruns"]}/{n} ({100*float(r["over_rate"]):.1f}%)',
                f'{float(r["mean_excess_late_ms"]):.2f}' if int(r["overruns"]) else "—"]
    heads = ["NDCG@10","实测效用","平均 ms","P95 ms","超时数与比例","超时平均超额 ms"]
    line("# AdaRetrieve 第 2、5、6 项补充实验")
    line("")
    line("本轮使用内置浏览器的 Colab Tesla T4。SciFact、NFCorpus、FiQA 共有 1,271 条留出查询。所有质量均按各域官方 test qrels 逐查询重算。原报告及工程包保留为第 5 版。")
    line("")
    line("## FiQA 14 ms")
    line("")
    table(["准入成本"]+heads,[[key]+metrics(guards["fiqa",14,key]) for key in ("v5","p50","p90","p95")])
    line("以下均为同一查询上的差值。括号内是逐查询配对 bootstrap 的 95% 区间（4,000 次重采样）。超时率的单位为百分点，负值代表超时减少。区间只反映查询抽样，不含跨运行时波动。")
    line("")
    delta = {(r["domain"],int(r["budget"]),r["guard"],r["metric"]):r for r in rows("paired-differences.csv")}
    def diff(key,metric):
        r=delta["fiqa",14,key,metric]
        s=100 if metric=="over_rate" else 1
        digits=1 if metric=="over_rate" else (2 if metric=="mean_ms" else 4)
        def fmt(k):return f"{float(r[k])*s:+.{digits}f}"
        return f'{fmt("delta")} [{fmt("ci95_low")}, {fmt("ci95_high")}]'
    table(["相对 p50","超时率差","NDCG@10 差","实测效用差","平均耗时差 ms"],
          [[key]+[diff(key,m) for m in ("over_rate","ndcg","utility","mean_ms")] for key in ("p90","p95")])
    if recorded_run:
        line("p95 相对 p50 少了 173 条超时查询，平均耗时低于 14 ms，仍有 54.8% 的查询超时。更早停止减少了超额罚分，同时损失了排序质量。p90、p95 的效用点估计均上升，两者的配对区间均跨过零。")
    line("")
    stage=rows("fiqa-stage-times.csv")
    line("下表按实际执行该动作的查询统计。完成多步的查询会出现在多行，末列为整条查询最终超时的条数。")
    line("")
    table(["规则","阶段","执行次数","阶段平均 ms","阶段 P95 ms","最终超时条数"],
          [[r["rule"],r["stage"],r["executions"],f'{float(r["mean_ms"]):.2f}',f'{float(r["p95_ms"]):.2f}',r["late_queries"]]
           for r in stage if r["stage"] in ("initial","dense20","dense64")])
    if recorded_run:
        line("FiQA 留出折从 SciFact、NFCorpus 估计的 dense20 成本为 p95 7.70 ms。在 FiQA 上，p95 规则实际执行 dense20 的 385 条查询中，该动作平均用时 11.40 ms，其中 354 条查询最终超时。源域分位数低估了这个留出域的实际稠密检索成本。")
    line("")
    line("## 源域校准与三折准入")
    line("")
    line("每域从训练部分按固定顺序预热 6 条，再对 96 个不同查询测量动作耗时。每折汇合其他两域的 192 个观测，对各动作分别求线性分位数。所有折的规则、源域、版本和散列先落盘，再打开测试标注。每轮采用相同的 BM25 复用、冻结的路径收益模型、成本惩罚和改道门槛。分位数决定初始路径、下一步、可改道后缀能否放入剩余预算。首个 BM25 步骤为必做。各分位数是单步耗时统计，无法直接给出整条路径的超时概率。")
    line("")
    times=rows("source-action-times.csv")
    actions=["initial","lex64","dense20","dense64","fuse","rerank8","rerank24"]
    def cost(d,key,a):
        return next(float(r["ms"]) for r in times if (r["target"],r["guard"],r["action"])==(d,key,a))
    table(["留出域","规则"]+actions,[[name,key]+[f"{cost(d,key,a):.3f}" for a in actions]
                                      for d,name in domain_names.items() for key in ("p50","p90","p95")])
    line("每次路由独立执行并实测，查询内随机交错四种规则和三档预算。效用逐查询计算为 NDCG@10 − 0.035 × t/B − 0.30 × max(0, t/B − 1)，包括全部计时和超额罚分。v5 为本场重测的原准入成本。")
    line("")
    table(["留出域","预算 ms","规则"]+heads,[[name,b,rule]+metrics(guards[d,b,rule])
          for d,name in domain_names.items() for b in bs[d] for rule in ("v5","p50","p90","p95")])
    line("## 检索和重排深度")
    line("")
    line("BM25 与稠密检索各最多返回 K 篇，RRF 合并去重。候选召回率为每个查询的候选并集覆盖的已标注相关文档比例，再按查询平均。基线为 192-token TinyBERT。每个 K/编码长度组合实测一个公共检索前缀，所有后缀均计入完整前缀耗时和各自实测的重排耗时。重排每次独立打分，同一查询的各条件随机排序。K=20 时并集最多 40 篇，重排 50 配置实际处理该查询可用的全部候选。")
    line("")
    def g(d,k,r,enc=192,ce=192,tier=1):return grid[d,bs[d][tier],k,enc,r,ce]
    table(["留出域","每路 K","候选召回率","RRF NDCG","重排 8 NDCG","重排 24 NDCG","重排 50 NDCG","RRF 平均 ms","重排 50 平均 ms","重排 50 实际平均篇数"],
          [[name,k,f'{float(g(d,k,0)["candidate_recall"]):.4f}']+
           [f'{float(g(d,k,r)["ndcg"]):.4f}' for r in (0,8,24,50)]+
           [f'{float(g(d,k,r)["mean_ms"]):.2f}' for r in (0,50)]+[f'{float(g(d,k,50)["scored_mean"]):.1f}']
           for d,name in domain_names.items() for k in (20,64,100)])
    if recorded_run:
        line("每路候选从 64 扩至 100 后，三个域的候选召回率均上升，固定 RRF 的 NDCG@10 差值均不超过 0.0032。每路 64 篇时，TinyBERT 重排深度从 24 增到 50，三个域的平均 NDCG 都下降，平均延迟增加 23.6–31.7 ms。")
    line("")
    line("## 输入长度与截断")
    line("")
    line("固定每路 K=64、重排前 24 篇，独立改变稠密编码器和 TinyBERT 重排器的上限。384-token 编码器重新编码全部文档并重建精确内积索引，建库时间单列。重排器的 token 统计按实际送入的查询与候选文档拼接并计入特殊符号；查询文档对统计固定使用 192-token 检索的 RRF 前 24 篇。")
    line("")
    lens=rows("truncation.csv")
    groups={"corpus_documents":"语料文档","heldout_queries":"测试查询","RRF64_top24_query_document_pairs":"前 24 查询文档对"}
    table(["留出域","输入","样本数","超过 192","超过 384"],
          [[domain_names[r["domain"]],groups[r["input"]],r["count"],
            f'{r["over192"]} ({100*float(r["rate192"]):.1f}%)',f'{r["over384"]} ({100*float(r["rate384"]):.1f}%)'] for r in lens])
    table(["留出域","编码器上限","重排器上限","NDCG@10","平均 ms","P95 ms","中档预算实测效用"],
          [[name,enc,ce]+[f'{float(g(d,64,24,enc,ce)[k]):.{4 if k in ("ndcg","utility") else 2}f}' for k in ("ndcg","mean_ms","p95_ms","utility")]
          for d,name in domain_names.items() for enc,ce in ((192,192),(192,384),(384,192),(384,384))])
    table(["留出域","384-token 重建文档数","离线编码和建索引秒"],
          [[domain_names[r["domain"]],r["reindex_documents"],f'{float(r["reindex_seconds"]):.1f}'] for r in lens if r["input"]=="corpus_documents"])
    if recorded_run:
        line("单独将重排器放宽至 384 tokens，SciFact 的 NDCG@10 增加 0.0478，查询配对 95% 区间为 [0.0260, 0.0702]，平均多花 6.12 ms。NFCorpus 提升 0.0083；FiQA 提升 0.0057，区间跨零。三个域里，只有 SciFact 的中档预算实测效用随之上升。")
    line("")
    line("单台 T4、每查询每条件一次，属于探索性诊断。所有预算下的 NDCG、实测效用、平均及 P95 耗时、超预算比例和幅度保存在同目录的 admission.csv、depth-length.csv。")
    line("")
    line("## 复核与复现")
    line("")
    line("observations.zip 包含每条查询的前十篇文档、准入路径、时间、源域采样、冻结清单和完整性校验码。将本目录放在第 5 版 project 目录的同级，运行 python budget-depth-t4/analyze.py 可用已校验的原始数据集重算质量、效用和区间，再运行 python budget-depth-t4/make_results.py 生成本页。重测 GPU 时打开 sensitivity-t4.ipynb，上传原来的工程 ZIP 和索引 ZIP，选择 Colab T4 后执行全部单元格。")
    (HERE/"results.md").write_text("\n".join(body)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
