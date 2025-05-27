from google_play_scraper import app, reviews, Sort
import pandas as pd
import time
from tqdm import tqdm
import re
import os
from datetime import datetime
import openai
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# 设置OpenAI API密钥
openai.api_key = os.environ.get("OPENAI_API_KEY")  # 从环境变量获取API密钥
def scrape_reviews_by_score(
        play_store_url: str,
        scores: list,  # 例如 [1] 或 [5] 或 [1,5]
        max_reviews: int = 300,  # 每种评分各爬多少条
        language: str = "en",
        country: str = "us",
        delay: float = 3.0
) -> pd.DataFrame:
    """爬取指定评分范围的评论"""
    # 从URL提取包名
    app_id = re.search(r'id=([a-zA-Z._]+)', play_store_url).group(1)
    all_reviews = []
    for score in scores:
        print(f"\n=== 开始爬取 {score} 星评论 ===")
        token = None
        pbar = tqdm(total=max_reviews, desc=f"⭐ {'★' * score} {score}-star reviews")
        while len([r for r in all_reviews if r['score'] == score]) < max_reviews:
            try:
                batch, token = reviews(
                    app_id,
                    lang=language,
                    country=country,
                    sort=Sort.NEWEST,
                    count=100,
                    continuation_token=token,
                    filter_score_with=score  # 关键修改：按评分过滤
                )
                if not batch:
                    print(f"\n⚠️ {score}星评论已爬完")
                    break
                all_reviews.extend(batch)
                pbar.update(len([r for r in batch if r['score'] == score]))
                time.sleep(delay)
            except Exception as e:
                if "HTTP 429" in str(e):
                    print("\n⏳ 被限速，等待10秒...")
                    time.sleep(10)
                else:
                    print(f"\n❌ 错误: {str(e)}")
                    break
    # 整理为DataFrame
    df = pd.DataFrame(all_reviews)[["userName", "score", "at", "content", "thumbsUpCount"]]
    return df[df['score'].isin(scores)]  # 确保只返回指定评分
def analyze_all_reviews(df, app_name):
    """分析所有评论并生成总体报告"""
    print("\n=== 开始进行AI总体分析 ===")
    # 准备评论文本
    all_reviews_text = ""
    for i, row in df.iterrows():
        all_reviews_text += f"评论 {i+1} (评分: {row['score']}星): {row['content']}\n\n"
    try:
        # 调用OpenAI API进行分析
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位专业的用户反馈分析师。"},
                {"role": "user", "content": f"""
                请分析以下关于应用"{app_name}"的Google Play商店评论，并提供详细分析报告。
                {all_reviews_text[:10000]}  # 限制长度，避免超出token限制
                请在分析中包含：
                1. 总体评价概述
                2. 最常见的用户问题/抱怨（按频率排序）
                3. 用户最欣赏的应用特性
                4. 高分评论中的共同点
                5. 低分评论中的共同点
                6. 给开发团队的改进建议（按优先级排序）
                7. 评论的情感分析（积极/消极/中立比例）
                请以结构化、专业的方式呈现分析结果。
                """}
            ]
        )
        analysis = response['choices'][0]['message']['content']
        print("✅ AI总体分析完成")
        return analysis
    except Exception as e:
        print(f"❌ AI分析失败: {str(e)}")
        return f"AI分析过程中发生错误: {str(e)}"
def compare_high_low_reviews(high_df, low_df, app_name):
    """比较高分和低分评论"""
    print("\n=== 开始比较高分与低分评论 ===")
    # 准备评论摘要
    high_reviews_sample = "\n".join([f"5星评论: {row['content'][:200]}..." for _, row in high_df.head(15).iterrows()])
    low_reviews_sample = "\n".join([f"1星评论: {row['content'][:200]}..." for _, row in low_df.head(15).iterrows()])
    try:
        # 调用OpenAI API进行对比分析
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位数据分析专家，擅长对比分析。"},
                {"role": "user", "content": f"""
                请对比分析"{app_name}"应用的高分(5星)和低分(1星)评论，找出关键差异。
                部分5星评论示例:
                {high_reviews_sample}
                部分1星评论示例:
                {low_reviews_sample}
                请进行以下分析：
                1. 高分与低分评论的核心差异
                2. 用户体验的关键分水岭（哪些因素决定了好评与差评）
                3. 低分评论中提到的具体问题和痛点
                4. 高分评论中特别赞赏的功能或特性
                5. 开发团队应该保留的优势和需要改进的方面
                请尽量具体，并提供可行的建议。
                """}
            ]
        )
        comparison = response['choices'][0]['message']['content']
        print("✅ 高低分对比分析完成")
        return comparison
    except Exception as e:
        print(f"❌ 对比分析失败: {str(e)}")
        return f"对比分析过程中发生错误: {str(e)}"
def get_key_action_items(overall_analysis, comparison_analysis):
    """提取关键行动项目"""
    print("\n=== 生成关键行动项目 ===")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位产品经理，擅长制定行动计划。"},
                {"role": "user", "content": f"""
                基于以下用户评论分析，请提取10个最关键的行动项目，按优先级排序：
                总体分析:
                {overall_analysis[:1500]}...
                高低分对比:
                {comparison_analysis[:1500]}...
                对于每个行动项目，请提供：
                1. 问题描述（简洁明了）
                2. 建议的解决方案
                3. 预期影响（对用户体验和评分的影响）
                4. 优先级（高/中/低）
                请以清晰的列表形式呈现这些行动项目。
                """}
            ]
        )
        action_items = response['choices'][0]['message']['content']
        print("✅ 关键行动项目生成完成")
        return action_items
    except Exception as e:
        print(f"❌ 行动项目生成失败: {str(e)}")
        return f"生成行动项目时发生错误: {str(e)}"
def generate_visualizations(df, output_dir):
    """生成数据可视化"""
    print("\n=== 生成数据可视化 ===")
    try:
        # 创建可视化目录
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        # 1. 评分分布饼图
        plt.figure(figsize=(10, 6))
        score_counts = df['score'].value_counts().sort_index()
        plt.pie(score_counts, labels=[f"{i}星" for i in score_counts.index], 
                autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(score_counts)))
        plt.title('评分分布')
        plt.savefig(vis_dir / "rating_distribution.png")
        plt.close()
        # 2. 评论时间趋势
        plt.figure(figsize=(12, 6))
        df['date'] = pd.to_datetime(df['at']).dt.date
        date_counts = df.groupby(['date', 'score']).size().unstack().fillna(0)
        date_counts.plot(kind='line', stacked=False)
        plt.title('评论时间趋势')
        plt.xlabel('日期')
        plt.ylabel('评论数量')
        plt.legend(title='评分')
        plt.tight_layout()
        plt.savefig(vis_dir / "review_trend.png")
        plt.close()
        # 3. 点赞数分布
        plt.figure(figsize=(10, 6))
        sns.histplot(df['thumbsUpCount'], bins=20)
        plt.title('评论点赞数分布')
        plt.xlabel('点赞数')
        plt.ylabel('频率')
        plt.savefig(vis_dir / "thumbs_up_distribution.png")
        plt.close()
        print(f"✅ 数据可视化已保存至 {vis_dir}")
    except Exception as e:
        print(f"❌ 生成可视化时发生错误: {str(e)}")
def create_report_html(app_name, app_info, df, overall_analysis, comparison, action_items, output_dir):
    """创建HTML分析报告"""
    print("\n=== 创建HTML分析报告 ===")
    try:
        # 基本统计数据
        total_reviews = len(df)
        avg_rating = df['score'].mean()
        rating_counts = df['score'].value_counts().sort_index().to_dict()
        # 创建HTML报告
        html_content = f"""<!DOCTYPE html>
        <html lang="zh">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{app_name} - 评论分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                h3 {{ color: #3498db; }}
                .stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }}
                .rating {{ font-size: 24px; font-weight: bold; color: #f39c12; }}
                .stars {{ font-size: 18px; }}
                .analysis {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }}
                .action-item {{ background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .priority-high {{ border-left: 4px solid #e74c3c; }}
                .priority-medium {{ border-left: 4px solid #f39c12; }}
                .priority-low {{ border-left: 4px solid #2ecc71; }}
                .review-sample {{ background: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .review-sample.positive {{ border-left: 3px solid #2ecc71; }}
                .review-sample.negative {{ border-left: 3px solid #e74c3c; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{app_name} - Google Play 评论分析报告</h1>
                <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class="stats">
                    <div class="stat-card">
                        <h3>总评论数</h3>
                        <div class="rating">{total_reviews}</div>
                    </div>
                    <div class="stat-card">
                        <h3>平均评分</h3>
                        <div class="rating">{avg_rating:.1f}</div>
                        <div class="stars">{'★' * int(round(avg_rating))}</div>
                    </div>
                    <div class="stat-card">
                        <h3>评分分布</h3>
                        <ul>
        """
        # 添加评分分布
        for score in sorted(rating_counts.keys(), reverse=True):
            count = rating_counts[score]
            percentage = (count / total_reviews) * 100
            html_content += f"<li>{score}星: {count}条 ({percentage:.1f}%)</li>\n"
        html_content += """
                        </ul>
                    </div>
                </div>
                <h2>数据可视化</h2>
                <div class="visuals">
                    <img src="visualizations/rating_distribution.png" alt="评分分布">
                    <img src="visualizations/review_trend.png" alt="评论趋势">
                    <img src="visualizations/thumbs_up_distribution.png" alt="点赞分布">
                </div>
                <h2>AI总体分析</h2>
                <div class="analysis">
        """
        # 添加总体分析（格式化处理）
        formatted_analysis = overall_analysis.replace('\n', '<br>')
        html_content += f"{formatted_analysis}</div>\n"
        html_content += """
                <h2>高分与低分评论对比</h2>
                <div class="analysis">
        """
        # 添加对比分析
        formatted_comparison = comparison.replace('\n', '<br>')
        html_content += f"{formatted_comparison}</div>\n"
        html_content += """
                <h2>关键行动项目</h2>
                <div class="action-items">
        """
        # 添加行动项目（简单处理，假设格式化后的文本）
        formatted_actions = action_items.replace('\n', '<br>')
        html_content += f"{formatted_actions}</div>\n"
        # 添加评论示例
        html_content += """
                <h2>评论示例</h2>
                <h3>正面评论示例</h3>
        """
        # 添加5个高分评论示例
        for _, row in df[df['score'] > 3].sort_values('thumbsUpCount', ascending=False).head(5).iterrows():
            html_content += f"""
                <div class="review-sample positive">
                    <p><strong>评分: {row['score']}星</strong> | 点赞: {row['thumbsUpCount']} | 日期: {pd.to_datetime(row['at']).strftime('%Y-%m-%d')}</p>
                    <p>{row['content']}</p>
                </div>
            """
        html_content += """
                <h3>负面评论示例</h3>
        """
        # 添加5个低分评论示例
        for _, row in df[df['score'] <= 2].sort_values('thumbsUpCount', ascending=False).head(5).iterrows():
            html_content += f"""
                <div class="review-sample negative">
                    <p><strong>评分: {row['score']}星</strong> | 点赞: {row['thumbsUpCount']} | 日期: {pd.to_datetime(row['at']).strftime('%Y-%m-%d')}</p>
                    <p>{row['content']}</p>
                </div>
            """
        html_content += """
            </div>
        </body>
        </html>
        """
        # 保存HTML报告
        html_path = output_dir / f"{app_name}_analysis_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML分析报告已保存至 {html_path}")
        return html_path
    except Exception as e:
        print(f"❌ 创建HTML报告时发生错误: {str(e)}")
        return None
def run_analysis_pipeline(app_url, scores=[1, 5], max_reviews_per_score=200, language="en", country="us"):
    """运行完整的分析流程"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建输出目录
    output_dir = Path(f"analysis_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    # 获取App信息
    try:
        app_id = re.search(r'id=([a-zA-Z._]+)', app_url).group(1)
        app_info = app(app_id, lang=language, country=country)
        app_name = app_info['title']
        app_name_safe = app_name.replace(" ", "_")
        print(f"\n==== 开始分析应用: {app_name} ====")
        # 保存应用基本信息
        with open(output_dir / f"{app_name_safe}_info.json", 'w', encoding='utf-8') as f:
            json.dump(app_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ 获取应用信息失败: {str(e)}")
        app_id = re.search(r'id=([a-zA-Z._]+)', app_url).group(1)
        app_name = app_id
        app_name_safe = app_id
        app_info = {"title": app_id, "appId": app_id}
    # 分别抓取各评分的评论
    all_reviews = []
    for score in scores:
        df_score = scrape_reviews_by_score(app_url, scores=[score], max_reviews=max_reviews_per_score, 
                                           language=language, country=country)
        all_reviews.append(df_score)
        # 保存各评分的评论
        csv_path = output_dir / f"{app_name_safe}_score_{score}_reviews.csv"
        df_score.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ {score}星评论已保存至 {csv_path} ({len(df_score)}条)")
    # 合并所有评论
    df_combined = pd.concat(all_reviews)
    combined_csv_path = output_dir / f"{app_name_safe}_all_reviews.csv"
    df_combined.to_csv(combined_csv_path, index=False, encoding='utf-8')
    print(f"✅ 所有评论已合并并保存至 {combined_csv_path} (共{len(df_combined)}条)")
    # 分离高分和低分评论（用于比较分析）
    high_df = df_combined[df_combined['score'] >= 4]
    low_df = df_combined[df_combined['score'] <= 2]
    # 生成数据可视化
    generate_visualizations(df_combined, output_dir)
    # 进行AI分析
    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
        # 1. 总体分析
        overall_analysis = analyze_all_reviews(df_combined, app_name)
        overall_path = output_dir / f"{app_name_safe}_overall_analysis.txt"
        with open(overall_path, 'w', encoding='utf-8') as f:
            f.write(overall_analysis)
        print(f"✅ 总体分析已保存至 {overall_path}")
        # 2. 高低分对比分析
        comparison = compare_high_low_reviews(high_df, low_df, app_name)
        comparison_path = output_dir / f"{app_name_safe}_high_low_comparison.txt"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write(comparison)
        print(f"✅ 高低分对比已保存至 {comparison_path}")
        # 3. 提取关键行动项目
        action_items = get_key_action_items(overall_analysis, comparison)
        actions_path = output_dir / f"{app_name_safe}_action_items.txt"
        with open(actions_path, 'w', encoding='utf-8') as f:
            f.write(action_items)
        print(f"✅ 关键行动项目已保存至 {actions_path}")
        # 4. 创建HTML报告
        html_report = create_report_html(app_name, app_info, df_combined, 
                                        overall_analysis, comparison, action_items, output_dir)
        # 5. 生成README.md
        with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(f"# {app_name} 评论分析报告\n\n")
            f.write(f"分析时间: {timestamp.replace('_', ' ')}\n\n")
            f.write(f"## 基本信息\n\n")
            f.write(f"- 应用名称: {app_name}\n")
            f.write(f"- 应用ID: {app_id}\n")
            f.write(f"- 总评论数: {len(df_combined)}\n")
            f.write(f"- 平均评分: {df_combined['score'].mean():.2f}/5\n\n")
            f.write(f"## 分析文件\n\n")
            f.write(f"-  if html_report else '#'})\n")
            f.write(f"- [总体分析](/{os.path.basename(overall_path)})\n")
            f.write(f"- [高低分对比](/{os.path.basename(comparison_path)})\n")
            f.write(f"- [关键行动项目](/{os.path.basename(actions_path)})\n")
            f.write(f"- [原始评论数据](/{os.path.basename(combined_csv_path)})\n\n")
            f.write(f"## 可视化\n\n")
            f.write(f"\n")
            f.write(f"![评论趋势](visualizations/review_trend.png)\n")
            f.write(f"![点赞分布](visualizations/thumbs_up_distribution.png)\n")
    else:
        print("\n⚠️ 未设置OPENAI_API_KEY环境变量，跳过AI分析步骤")
        # 创建简单的统计分析报告
        stats_report = f"""
        # {app_name} 评论基本统计
        ## 汇总统计
        - 总评论数: {len(df_combined)}
        - 平均评分: {df_combined['score'].mean():.2f}/5
        ## 评分分布
        {df_combined['score'].value_counts().sort_index().to_string()}
        ## 点赞统计
        - 平均点赞数: {df_combined['thumbsUpCount'].mean():.2f}
        - 最高点赞数: {df_combined['thumbsUpCount'].max()}
        ## 时间分布
        {df_combined.groupby(pd.to_datetime(df_combined['at']).dt.date).size().sort_index().tail(10).to_string()}
        注: 由于未设置OpenAI API密钥，无法进行AI内容分析。请设置环境变量OPENAI_API_KEY后重新运行以获取完整分析。
        """
        stats_path = output_dir / f"{app_name_safe}_basic_stats.md"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(stats_report)
        print(f"✅ 基本统计报告已保存至 {stats_path}")
    print(f"\n✅✅✅ 分析完成! 所有结果已保存至 {output_dir} 目录")
    return output_dir
if __name__ == "__main__":
    # 配置参数
    app_url = "https://play.google.com/store/apps/details?id=screw.puzzle.nuts.bolts.pin.wood.games"
    # 运行分析流程
    analysis_dir = run_analysis_pipeline(
        app_url=app_url,
        scores=[1, 5],  # 爬取1星和5星评论
        max_reviews_per_score=200,  # 每种评分最多爬取200条
        language="en",  # 语言设置为英文
        country="us"    # 国家设置为美国
    )
    print(f"\n分析报告已生成在 {analysis_dir} 目录，请查看该目录中的HTML报告和其他分析文件。")
