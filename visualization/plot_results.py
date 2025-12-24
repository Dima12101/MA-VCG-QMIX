import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Установить стиль
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

class ResultsVisualizer:
    """Класс для визуализации результатов экспериментов"""
    
    def __init__(self, results_dir: str = 'experiments/results'):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_social_welfare(self, scenario_name: str = 'scenario_1'):
        """График эволюции социального благосустояния"""
        csv_path = self.results_dir / scenario_name / f'{scenario_name}_episode_0.csv'
        
        if not csv_path.exists():
            print(f"Файл {csv_path} не найден")
            return
        
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['time'], df['social_welfare'], linewidth=2, label='Social Welfare')
        
        # Добавить линию тренда
        z = np.polyfit(df['time'], df['social_welfare'], 3)
        p = np.poly1d(z)
        plt.plot(df['time'], p(df['time']), '--', alpha=0.5, label='Trend')
        
        plt.xlabel('Time (steps)', fontsize=12)
        plt.ylabel('Social Welfare', fontsize=12)
        plt.title('Evolution of Social Welfare', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        save_path = self.plots_dir / f'{scenario_name}_social_welfare.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def plot_fairness_metrics(self, scenario_name: str = 'scenario_1'):
        """График метрик справедливости"""
        csv_path = self.results_dir / scenario_name / f'{scenario_name}_episode_0.csv'
        
        df = pd.read_csv(csv_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Коэффициент Джини
        ax1.plot(df['time'], df['gini_payment'], linewidth=2, color='red')
        ax1.fill_between(df['time'], df['gini_payment'], alpha=0.3, color='red')
        ax1.set_xlabel('Time (steps)', fontsize=11)
        ax1.set_ylabel('Gini Coefficient', fontsize=11)
        ax1.set_title('Payment Fairness (Gini)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.3, color='green', linestyle='--', label='Good fairness threshold')
        ax1.legend()
        
        # Индекс справедливости
        ax2.plot(df['time'], df['fairness_index'], linewidth=2, color='blue')
        ax2.fill_between(df['time'], df['fairness_index'], alpha=0.3, color='blue')
        ax2.set_xlabel('Time (steps)', fontsize=11)
        ax2.set_ylabel('Fairness Index', fontsize=11)
        ax2.set_title('Resource Allocation Fairness', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.85, color='green', linestyle='--', label='Good fairness threshold')
        ax2.legend()
        
        plt.tight_layout()
        save_path = self.plots_dir / f'{scenario_name}_fairness_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def plot_load_distribution(self, scenario_name: str = 'scenario_1'):
        """График распределения нагрузки по узлам"""
        csv_path = self.results_dir / scenario_name / f'{scenario_name}_episode_0.csv'
        
        df = pd.read_csv(csv_path)
        
        # Найти столбцы с нагрузкой узлов
        load_cols = [col for col in df.columns if col.startswith('load_node')]
        
        plt.figure(figsize=(12, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, col in enumerate(load_cols):
            plt.plot(df['time'], df[col], label=f'Node {i}', linewidth=2, color=colors[i % len(colors)])
        
        plt.xlabel('Time (steps)', fontsize=12)
        plt.ylabel('Node Load', fontsize=12)
        plt.title('Load Distribution Across Edge Nodes', fontsize=14)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        save_path = self.plots_dir / f'{scenario_name}_load_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def plot_acceptance_rate(self, scenario_name: str = 'scenario_1'):
        """График процента принятых задач"""
        csv_path = self.results_dir / scenario_name / f'{scenario_name}_episode_0.csv'
        
        df = pd.read_csv(csv_path)
        
        # Вычислить процент принятых задач
        total = df['accepted_tasks'] + df['rejected_tasks']
        acceptance_rate = (df['accepted_tasks'] / total * 100).fillna(0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['time'], acceptance_rate, linewidth=2, label='Acceptance Rate', color='green')
        plt.fill_between(df['time'], acceptance_rate, alpha=0.3, color='green')
        
        # Скользящее среднее
        window = 50
        ma = acceptance_rate.rolling(window).mean()
        plt.plot(df['time'], ma, '--', linewidth=2, label=f'MA (window={window})', color='darkgreen')
        
        plt.xlabel('Time (steps)', fontsize=12)
        plt.ylabel('Acceptance Rate (%)', fontsize=12)
        plt.title('Task Acceptance Rate Over Time', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        
        save_path = self.plots_dir / f'{scenario_name}_acceptance_rate.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def plot_latency_distribution(self, scenario_name: str = 'scenario_1'):
        """Гистограмма распределения задержек"""
        csv_path = self.results_dir / scenario_name / f'{scenario_name}_episode_0.csv'
        
        df = pd.read_csv(csv_path)
        latencies = df['avg_latency'].dropna()
        
        plt.figure(figsize=(12, 6))
        plt.hist(latencies, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(latencies.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {latencies.mean():.1f} ms')
        plt.axvline(latencies.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {latencies.median():.1f} ms')
        
        plt.xlabel('Latency (ms)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Latency Distribution', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = self.plots_dir / f'{scenario_name}_latency_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()
    
    def plot_comparison_scenarios(self):
        """Сравнение всех сценариев"""
        scenarios = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4']
        summaries = []
        
        for scenario in scenarios:
            summary_path = self.results_dir / scenario / 'summary.csv'
            if summary_path.exists():
                summary = pd.read_csv(summary_path)
                summaries.append(summary)
        
        if not summaries:
            print("Не найдены файлы summary.csv")
            return
        
        df = pd.concat(summaries, ignore_index=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Acceptance Rate
        axes[0, 0].bar(df['scenario'], df['avg_acceptance_rate'], color='green', alpha=0.7)
        axes[0, 0].set_ylabel('Acceptance Rate (%)', fontsize=11)
        axes[0, 0].set_title('Task Acceptance Rate by Scenario', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Social Welfare
        axes[0, 1].bar(df['scenario'], df['avg_sw'], color='blue', alpha=0.7)
        axes[0, 1].set_ylabel('Social Welfare', fontsize=11)
        axes[0, 1].set_title('Average Social Welfare by Scenario', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Fairness (Gini)
        axes[1, 0].bar(df['scenario'], df['avg_gini'], color='red', alpha=0.7)
        axes[1, 0].set_ylabel('Gini Coefficient', fontsize=11)
        axes[1, 0].set_title('Payment Fairness (Gini) by Scenario', fontsize=12)
        axes[1, 0].axhline(y=0.3, color='green', linestyle='--', label='Good fairness')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Summary metrics
        axes[1, 1].axis('off')
        summary_text = "Summary Statistics\n\n"
        for col in ['avg_acceptance_rate', 'avg_sw', 'avg_gini']:
            summary_text += f"{col}:\n"
            summary_text += f"  Mean: {df[col].mean():.2f}\n"
            summary_text += f"  Std: {df[col].std():.2f}\n\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.plots_dir / 'scenario_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
        plt.close()

if __name__ == '__main__':
    viz = ResultsVisualizer()
    
    # Построить графики для Сценария 1
    scenario = 'scenario_1'
    print(f"Построение графиков для {scenario}...")
    
    viz.plot_social_welfare(scenario)
    viz.plot_fairness_metrics(scenario)
    viz.plot_load_distribution(scenario)
    viz.plot_acceptance_rate(scenario)
    viz.plot_latency_distribution(scenario)
    
    # Сравнение сценариев
    viz.plot_comparison_scenarios()
    
    print("\nВсе графики построены!")
