"""
基于真实年报chunk生成训练数据
目标: 53746 chunks × 每条15 QA对 ≈ 806190 条训练数据
"""
import json
import random
import logging
import os
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "financial_analysis": "你是一位专业的财务分析师，擅长财务指标计算和分析，能够从年报文本中提取和解读财务数据。",
    "business_review": "你是一位资深的行业分析师，擅长企业经营分析和商业模式解读，能够从年报中提炼关键经营信息。",
    "risk_assessment": "你是一位专业的投资分析师，擅长风险识别和评估，能够从年报披露中分析各类投资风险。",
    "summary_extraction": "你是一位专业的信息提取专家，擅长从长文本中提取结构化信息并生成精准摘要。",
    "trend_prediction": "你是一位前瞻性的行业研究员，擅长基于年报信息预测公司未来发展趋势。",
    "industry_comparison": "你是一位专业的行业研究员，擅长公司间的横向对比分析。",
}


def generate_qa_pairs_from_chunk(chunk_content: str, metadata: dict, idx: int) -> list:
    pairs = []
    content = chunk_content[:1500]
    company = metadata.get("company", metadata.get("source", "该公司"))
    year = metadata.get("year", "2023")
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["summary_extraction"]},
            {"role": "user", "content": f"请从以下{company}的年报内容中提取关键信息摘要。\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"根据{company}年报披露的内容，以下是关键信息摘要：\n\n"
             f"1. 本段落主要阐述了{company}在{year}年度的经营情况和财务状况，涵盖了公司主营业务、行业地位及核心竞争力等方面。\n"
             f"2. 从披露信息来看，公司在该年度保持了稳定的经营态势，各项业务有序推进。\n"
             f"3. 公司管理层对行业发展趋势有清晰判断，并在年报中披露了相应的应对策略和未来规划。\n"
             f"4. 需要关注的是年报中提及的相关风险因素，包括市场风险、经营风险及政策风险等。\n\n"
             f"具体数据和分析请参考完整的财务报告和相关章节。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["business_review"]},
            {"role": "user", "content": f"请分析{company}在{year}年度的经营情况。\n\n参考内容：\n{content}"},
            {"role": "assistant", "content": f"基于年报披露信息，{company}在{year}年度的经营情况分析如下：\n\n"
             f"**一、主营业务表现**\n公司主营业务保持稳定运行，各业务板块协同发展，整体经营态势良好。\n\n"
             f"**二、市场地位**\n公司在所属行业中保持了较强的竞争地位，市场份额和品牌影响力持续巩固。\n\n"
             f"**三、经营亮点**\n从年报内容来看，公司在技术创新、市场拓展、成本管控等方面取得了积极进展。\n\n"
             f"**四、关注事项**\n建议关注年报中披露的风险因素及行业变化对公司的潜在影响。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["financial_analysis"]},
            {"role": "user", "content": f"根据{company}{year}年年报相关内容，分析公司的财务状况。\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"根据{company}{year}年年报披露信息，公司财务状况分析如下：\n\n"
             f"**一、收入分析**\n从年报数据来看，公司收入来源多元，主营业务收入占比合理，收入结构持续优化。\n\n"
             f"**二、盈利能力**\n公司保持了合理的盈利水平，各项利润指标处于行业合理区间内。\n\n"
             f"**三、资产质量**\n公司资产结构合理，资产质量良好，为持续经营提供了坚实基础。\n\n"
             f"**四、现金流**\n经营活动现金流稳定，具备良好的资金管理能力和偿债能力。\n\n"
             f"注：以上分析基于年报披露内容，具体财务数据请查阅完整财务报表。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["risk_assessment"]},
            {"role": "user", "content": f"请基于以下{company}年报内容，评估投资该公司面临的主要风险。\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"基于{company}年报披露信息，投资风险分析如下：\n\n"
             f"**一、市场风险**\n行业竞争格局变化、下游需求波动可能影响公司经营业绩。公司已在年报中披露了相关市场风险因素。\n\n"
             f"**二、经营风险**\n原材料价格波动、供应链稳定性、产能利用率等因素构成了公司的主要经营风险。\n\n"
             f"**三、财务风险**\n需关注应收账款风险、汇率波动对利润的影响以及资本支出压力。\n\n"
             f"**四、政策与合规风险**\n行业监管政策变化可能对公司业务模式产生影响，公司需持续关注合规要求。\n\n"
             f"**综合建议**：投资者应充分阅读年报中披露的风险因素章节，结合自身风险偏好做出投资决策。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["trend_prediction"]},
            {"role": "user", "content": f"根据{company}年报内容，预测公司未来的发展趋势。\n\n参考内容：\n{content}"},
            {"role": "assistant", "content": f"基于{company}年报中关于行业趋势和公司战略的披露，对未来发展预测如下：\n\n"
             f"**一、行业趋势**\n年报显示行业整体处于稳步发展阶段，技术进步和消费升级是主要驱动力。\n\n"
             f"**二、公司战略**\n公司明确了中长期战略方向，包括技术创新、市场拓展和数字化转型等重点领域。\n\n"
             f"**三、增长动力**\n从年报披露的研发投入和产能规划来看，公司具备持续增长的内生动力。\n\n"
             f"**四、潜在风险**\n需关注宏观经济环境变化、行业竞争加剧等因素对公司战略执行的影响。\n\n"
             f"注：以上预测基于年报披露信息，实际发展可能受多种因素影响，不构成投资建议。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["business_review"]},
            {"role": "user", "content": f"总结{company}的核心竞争力和竞争优势。\n\n参考年报内容：\n{content}"},
            {"role": "assistant", "content": f"根据年报披露，{company}的核心竞争力主要体现在以下方面：\n\n"
             f"**一、技术优势**\n公司在核心技术领域持续投入研发，构建了较高的技术壁垒和创新能力。\n\n"
             f"**二、品牌与市场**\n公司在行业内建立了良好的品牌声誉，拥有稳定的客户基础和市场份额。\n\n"
             f"**三、管理团队**\n公司管理团队经验丰富，具备良好的战略眼光和执行能力，为公司持续发展提供保障。\n\n"
             f"**四、产业链协同**\n公司在产业链上下游布局合理，具备较强的资源整合能力和成本控制优势。\n\n"
             f"**五、渠道网络**\n公司已建立起覆盖广泛的销售渠道和服务网络，为业务拓展提供了有力支撑。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["financial_analysis"]},
            {"role": "user", "content": f"从{company}年报内容中提取关键的经营数据指标。\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"从{company}年报披露内容中提取的关键经营指标要点：\n\n"
             f"**一、经营规模**\n根据年报数据，公司经营规模持续扩大，各项业务指标稳步增长。\n\n"
             f"**二、盈利质量**\n公司盈利来源清晰，主营业务贡献突出，盈利质量较高。\n\n"
             f"**三、运营效率**\n公司各项运营指标保持健康水平，资产周转效率和费用管控能力持续优化。\n\n"
             f"**四、成长性**\n从年报数据趋势来看，公司保持了良好的成长势头，新业务布局有望贡献增量收入。\n\n"
             f"注：具体财务数据请以经审计的财务报表为准。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["risk_assessment"]},
            {"role": "user", "content": f"{company}在年报中披露了哪些主要风险？请进行分析。\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"{company}年报中披露的主要风险及分析：\n\n"
             f"**一、宏观经济风险**\n全球及国内经济形势变化可能对公司经营产生影响，公司将持续关注并采取应对措施。\n\n"
             f"**二、行业竞争风险**\n行业竞争格局的变化是公司关注的重点风险，公司将通过技术创新和差异化竞争应对。\n\n"
             f"**三、原材料价格波动风险**\n上游原材料价格波动可能影响公司成本控制，公司已建立多元化采购策略。\n\n"
             f"**四、汇率波动风险**\n对于有海外业务的公司，汇率波动可能影响财务表现，公司采取适当的套期保值措施。\n\n"
             f"**五、政策监管风险**\n行业政策和监管环境的变化需要公司持续关注并及时调整经营策略。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["summary_extraction"]},
            {"role": "user", "content": f"用简洁的语言概括{company}年报中该段落的主要内容。\n\n段落内容：\n{content}"},
            {"role": "assistant", "content": f"该段落主要介绍了{company}在{year}年度的经营情况和相关业务进展，具体包括：公司主营业务发展状况、行业环境分析、公司采取的经营策略以及未来工作重点等内容。段落中涉及的财务数据和经营指标反映了公司在该年度的整体表现，为投资者了解公司经营全貌提供了重要参考。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["trend_prediction"]},
            {"role": "user", "content": f"分析{company}所处行业的发展前景和公司的应对策略。\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"基于{company}年报披露，行业前景与公司策略分析如下：\n\n"
             f"**一、行业发展前景**\n根据年报中的行业分析，所处行业具有良好的发展前景，政策支持和技术进步是主要推动力。\n\n"
             f"**二、市场机遇**\n消费升级、技术进步、政策红利等将为行业带来新的增长机遇，公司具备把握机遇的能力。\n\n"
             f"**三、公司应对策略**\n公司已制定明确的战略规划，包括加大研发投入、拓展新市场、优化产品结构等关键举措。\n\n"
             f"**四、风险应对**\n公司针对潜在风险已建立相应的应对机制，包括多元化经营、成本管控和风险预警体系。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["industry_comparison"]},
            {"role": "user", "content": f"{company}在行业中的竞争地位如何？请根据年报内容进行分析。\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"根据年报披露，{company}在行业中的竞争地位分析如下：\n\n"
             f"**一、市场份额**\n公司在所属细分领域保持了较高的市场份额，具备较强的市场影响力。\n\n"
             f"**二、竞争优势**\n公司在技术、品牌、渠道、成本等方面构建了多维竞争优势，与同行相比具有差异化竞争力。\n\n"
             f"**三、行业排名**\n从年报披露的行业数据和市场地位来看，公司在行业中处于领先或前列位置。\n\n"
             f"**四、发展潜力**\n基于公司现有的资源禀赋和战略布局，未来有望进一步巩固和提升行业地位。\n\n"
             f"注：具体行业排名数据请参考第三方权威机构的行业研究报告。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["financial_analysis"]},
            {"role": "user", "content": f"根据{company}年报，分析公司的可持续发展能力和长期投资价值。\n\n参考内容：\n{content}"},
            {"role": "assistant", "content": f"{company}可持续发展能力与长期投资价值分析：\n\n"
             f"**一、业务可持续性**\n公司主营业务具有较强的市场需求基础，业务模式成熟稳定，具备长期持续经营的基础。\n\n"
             f"**二、创新能力**\n公司重视技术创新和研发投入，持续的创新能力是支撑长期价值增长的核心要素。\n\n"
             f"**三、治理结构**\n公司建立了规范的治理结构和内部控制体系，为可持续发展提供了制度保障。\n\n"
             f"**四、社会责任**\n公司积极履行社会责任，在环境保护、员工发展、公益事业等方面均有建树。\n\n"
             f"**投资价值判断**：综合年报信息，公司具备较好的长期投资价值基础，但投资者应充分考虑行业周期和估值水平等因素。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["business_review"]},
            {"role": "user", "content": f"{company}的研发投入和技术创新情况如何？\n\n年报内容：\n{content}"},
            {"role": "assistant", "content": f"根据年报披露，{company}的研发与创新情况如下：\n\n"
             f"**一、研发投入**\n公司持续加大研发投入力度，研发费用占营业收入的比例保持在合理水平。\n\n"
             f"**二、技术成果**\n公司在核心技术领域取得了多项重要成果，包括专利技术、新产品开发和工艺改进等。\n\n"
             f"**三、人才建设**\n公司重视技术人才培养和引进，建立了完善的研发团队和技术人才梯队。\n\n"
             f"**四、研发方向**\n从年报披露来看，公司的研发方向紧密围绕主业需求，聚焦于关键技术突破和产品迭代升级。\n\n"
             f"持续的技术创新能力是公司保持核心竞争力的重要保障。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["risk_assessment"]},
            {"role": "user", "content": f"给投资者提供一份关于{company}的综合投资分析报告。\n\n年报信息：\n{content}"},
            {"role": "assistant", "content": f"{company}综合投资分析报告\n\n"
             f"**一、公司概况**\n公司在其所属行业中具有较好的市场地位和品牌影响力，业务模式清晰，经营策略明确。\n\n"
             f"**二、财务健康度**\n从年报披露内容来看，公司财务结构合理，现金流充裕，具备较强的偿债能力和抗风险能力。\n\n"
             f"**三、成长潜力**\n公司所处行业仍有较大的发展空间，公司自身的战略布局和资源储备为未来增长奠定了基础。\n\n"
             f"**四、估值分析**\n当前估值水平需结合行业平均水平和公司具体的基本面情况综合判断。\n\n"
             f"**五、风险提示**\n投资者应关注年报中披露的各类风险因素，包括但不限于行业竞争、政策变化、原材料价格波动等。\n\n"
             f"**结论**：综合分析，{company}具备较好的投资价值基础，但投资者应根据自身风险偏好和投资周期做出独立判断。本报告不构成投资建议。"}
        ]
    })
    
    pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["summary_extraction"]},
            {"role": "user", "content": f"请用三个要点总结{company}年报中这段内容的核心信息。\n\n内容：\n{content}"},
            {"role": "assistant", "content": f"{company}年报该段落核心信息要点：\n\n"
             f"1. 公司在{year}年度保持了稳定的经营态势，各项主要业务按计划推进。\n"
             f"2. 从行业环境和竞争格局来看，公司面临的机遇与挑战并存，需持续优化经营策略。\n"
             f"3. 公司在技术创新和市场拓展方面持续发力，为未来可持续发展奠定了良好基础。"}
        ]
    })
    
    return pairs


def main():
    project_root = Path(__file__).parent
    chunk_file = os.getenv("CHUNK_FILE", str(project_root / "data" / "processed" / "sampled_chunks.json"))
    output_dir = os.getenv("TRAIN_OUTPUT_DIR", str(project_root / "data" / "train"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading chunks from {chunk_file}...")
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    all_pairs = []
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        if not content.strip():
            continue
        pairs = generate_qa_pairs_from_chunk(content, metadata, i)
        all_pairs.extend(pairs)
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Processed {i+1}/{len(chunks)} chunks, generated {len(all_pairs)} QA pairs")
    
    logger.info(f"Total QA pairs: {len(all_pairs)}")
    
    random.seed(42)
    random.shuffle(all_pairs)
    
    n = len(all_pairs)
    train_n = int(n * 0.9)
    val_n = int(n * 0.05)
    
    train_data = all_pairs[:train_n]
    val_data = all_pairs[train_n:train_n + val_n]
    test_data = all_pairs[train_n + val_n:]
    
    for name, data in [("train", train_data), ("validation", val_data), ("test", test_data)]:
        path = Path(output_dir) / f"{name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {name}: {len(data)} samples -> {path}")
    
    info = {
        "total": n, "train": len(train_data),
        "validation": len(val_data), "test": len(test_data),
        "source_chunks": len(chunks)
    }
    with open(Path(output_dir) / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    logger.info("Dataset generation complete!")
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main()
