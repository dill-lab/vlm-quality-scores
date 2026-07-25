import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from tqdm import tqdm
import warnings

# Set style
# sns.set_theme(style="whitegrid")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def compute_ece(y_true, y_prob, n_bins=10):
    bin_num_positives = []
    bin_mean_probs = []
    bin_num_instances = []
    
    # Ensure y_prob is within [0, 1]
    y_prob = np.clip(y_prob, 0, 1)
    
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins if i != n_bins - 1 else 1.01
        idx = np.where((y_prob >= lower) & (y_prob < upper))[0]
        if len(idx) > 0:
            bin_num_positives.append(np.mean(y_true[idx]))
            bin_mean_probs.append(np.mean(y_prob[idx]))
            bin_num_instances.append(len(idx))
        else:
            bin_num_positives.append(0)
            bin_mean_probs.append(0)
            bin_num_instances.append(0)
            
    total_instances = np.sum(bin_num_instances)
    if total_instances == 0:
        return None
        
    ece = sum(count * abs(pos - prob)
              for pos, prob, count in zip(bin_num_positives, bin_mean_probs, bin_num_instances))
    return ece / total_instances

def compute_discriminability(y_true, y_score):
    # Discriminability: Difference of means (Correct - Incorrect)
    # Significance: Student's t-test (pooled variance)
    
    scores_correct = y_score[y_true == 1]
    scores_incorrect = y_score[y_true == 0]
    
    if len(scores_correct) == 0 or len(scores_incorrect) == 0:
        return 0.0, 1.0

    # Difference of means
    disc = np.mean(scores_correct) - np.mean(scores_incorrect)
    
    # Check for constant inputs to avoid warnings
    if np.std(scores_correct) < 1e-9 and np.std(scores_incorrect) < 1e-9:
        if abs(np.mean(scores_correct) - np.mean(scores_incorrect)) < 1e-9:
            p_val = 1.0
        else:
            # Means differ but variances are 0 -> t-stat is infinite
            p_val = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # User specified equal_var=True (Student's t-test)
            t_stat, p_val = stats.ttest_ind(scores_correct, scores_incorrect, equal_var=True)
            
    return disc, p_val

def get_bin_counts(y_true, y_prob, n_bins=10):
    y_prob = np.clip(y_prob, 0, 1)
    bin_num_instances = []
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins if i != n_bins - 1 else 1.01
        idx = np.where((y_prob >= lower) & (y_prob < upper))[0]
        bin_num_instances.append(len(idx))
    return bin_num_instances

def plot_calibration_curve(y_true, y_prob, ax, title, vmax, cmap_name='crest'):
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_num_positives = []
    bin_mean_probs = []
    bin_num_instances = []
    bin_centers = []

    y_prob = np.clip(y_prob, 0, 1)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        # For the last bin, include the right edge
        if i == n_bins - 1:
            idx = np.where((y_prob >= lower) & (y_prob <= upper))[0]
        else:
            idx = np.where((y_prob >= lower) & (y_prob < upper))[0]
        if len(idx) > 0:
            bin_num_positives.append(np.mean(y_true[idx]))
            bin_mean_probs.append(np.mean(y_prob[idx]))
            bin_num_instances.append(len(idx))
        else:
            bin_num_positives.append(0)
            bin_mean_probs.append(0)
            bin_num_instances.append(0)
        bin_centers.append((i + 0.5) / n_bins)

    total_instances = np.sum(bin_num_instances)
    ece = sum(count * abs(pos - prob)
              for pos, prob, count in zip(bin_num_positives, bin_mean_probs, bin_num_instances))
    ece = ece / total_instances if total_instances > 0 else None

    # Use sns.barplot (seaborn's palette + saturation) to match reference styling
    df = pd.DataFrame({
        'bin_num_positives': bin_num_positives,
        'bin_centers': bin_centers,
        'bin_num_instances': bin_num_instances,
    })
    sns.barplot(
        x='bin_centers', y='bin_num_positives', data=df, ax=ax,
        hue='bin_num_instances', palette=cmap_name,
        edgecolor='black', linewidth=0.8, width=1,
        hue_norm=(0, vmax), legend=False,
    )

    # Diagonal (category positions run 0..n_bins-1)
    ax.plot([-0.5, n_bins - 0.5], [0, 1], 'k--', linewidth=1, alpha=0.8)

    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Quality Score', fontsize=17)
    ax.set_ylabel('')  # set externally only on leftmost subplot of each row

    # Ticks at bin edges (mapped to category-axis positions)
    ax.set_xticks(np.arange(-0.5, n_bins + 0.5, 1))
    ax.set_xticklabels([f"{i/n_bins:.1f}" for i in range(n_bins + 1)], fontsize=9.5)
    ax.tick_params(axis='y', labelsize=11)

    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    title_str = f"{title}\nECE = {ece:.3f}" if ece is not None else title
    ax.set_title(title_str, fontsize=17.5)

    return ece

def analyze_dataset(dataset_name, models, output_dir):
    results = []
    
    for model in models:
        csv_path = os.path.join('model_outputs', dataset_name, f'{model}.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        print(f"Processing {dataset_name} - {model}...")
        df = pd.read_csv(csv_path)
        
        if 'is_correct' not in df.columns:
            print(f"  'is_correct' column missing in {model}. Skipping.")
            continue
            
        df = df.dropna(subset=['is_correct'])

        # Use Soft Labels (Raw values) for ECE
        y_true_soft = df["is_correct"].values
        # Use Binary Labels for Discriminability
        y_true_binary = (df["is_correct"] >= 0.5).astype(int).values
        
        metrics_map = {
            'Simulatability': 'entail_prob',
            'Informativeness': 'informative',
            'Plausibility': 'commonsense_plausibility',
            'Visual Fidelity': 'visual_fidelity',
            'Contrastiveness': 'contrastiveness_score'
        }
        
        if 'visual_fidelity' in df.columns and 'contrastiveness_score' in df.columns:
            vf = df['visual_fidelity'].fillna(0)
            cs = df['contrastiveness_score'].fillna(0)
            df['Avg(VF, Contr.)'] = (vf + cs) / 2
            df['Prod(VF, Contr.)'] = vf * cs
            df['Min(VF, Contr.)'] = np.minimum(vf, cs)
            
            metrics_map.update({
                'Avg(VF, Contr.)': 'Avg(VF, Contr.)',
                'Prod(VF, Contr.)': 'Prod(VF, Contr.)',
                'Min(VF, Contr.)': 'Min(VF, Contr.)'
            })
            
        plot_metrics = ['Simulatability', 'Informativeness', 'Plausibility', 
                        'Visual Fidelity', 'Contrastiveness', 'Avg(VF, Contr.)']
        
        # First pass: collect data
        # Fixed max count for color scaling as requested
        global_max_count = 500
        valid_metrics_data = {}
        
        for metric_name in plot_metrics:
            col_name = metrics_map.get(metric_name)
            if col_name and col_name in df.columns:
                valid_mask = df[col_name].notna()
                y_prob = df.loc[valid_mask, col_name].astype(float).values
                y_true_s = y_true_soft[valid_mask]
                y_true_b = y_true_binary[valid_mask]
                
                if len(y_prob) > 0:
                    # counts = get_bin_counts(y_true_curr, y_prob)
                    # global_max_count = max(global_max_count, max(counts))
                    valid_metrics_data[metric_name] = (y_true_s, y_true_b, y_prob)

        # Second pass: Plot
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        axes = axes.flatten()
        
        for i, metric_name in enumerate(plot_metrics):
            ax = axes[i]
            if metric_name in valid_metrics_data:
                y_true_s, y_true_b, y_prob = valid_metrics_data[metric_name]
                # ECE uses Soft Labels
                ece = plot_calibration_curve(y_true_s, y_prob, ax, metric_name, vmax=global_max_count)
                # Discriminability uses Binary Labels
                disc, p_val = compute_discriminability(y_true_b, y_prob)
                
                results.append({
                    'Dataset': dataset_name,
                    'Model': model,
                    'Metric': metric_name,
                    'ECE': ece,
                    'Discriminability': disc,
                    'P-Value': p_val
                })
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Shared y-axis label on leftmost subplot of each row
        for leftmost_idx in (0, 3):
            axes[leftmost_idx].set_ylabel('Prediction Accuracy', fontsize=17)

        # Add shared colorbar
        norm = mpl.colors.Normalize(vmin=0, vmax=global_max_count)
        sm = plt.cm.ScalarMappable(cmap='crest', norm=norm)
        sm.set_array([])
        
        # Add colorbar to the right of the figure
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8]) # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=12)
        
        
        # Adjust layout
        plt.subplots_adjust(right=0.88, hspace=0.6, wspace=0.25)
        
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'{dataset_name}_{model}_calibration.pdf'), bbox_inches='tight')
        plt.close(fig)
        
        # Calculate for non-plotted metrics
        for metric_name in ['Prod(VF, Contr.)', 'Min(VF, Contr.)']:
            col_name = metrics_map.get(metric_name)
            if col_name and col_name in df.columns:
                 valid_mask = df[col_name].notna()
                 y_prob = df.loc[valid_mask, col_name].astype(float).values
                 y_true_curr = y_true_b[valid_mask]
                 if len(y_prob) > 0:
                    ece = compute_ece(y_true_curr, y_prob)
                    disc, p_val = compute_discriminability(y_true_curr, y_prob)
                    results.append({
                        'Dataset': dataset_name,
                        'Model': model,
                        'Metric': metric_name,
                        'ECE': ece,
                        'Discriminability': disc,
                        'P-Value': p_val
                    })

        # Load uncertainty baselines if available
        baseline_map = {
            'llava-v1.5-7b': 'llava-v1.5-7b.csv',
            'qwen2.5-vl-7b-instruct': 'qwen2.5-vl-7b.csv',
            'gpt-4o-2024-05-13': 'gpt-4o.csv',
        }
        baseline_filename = baseline_map.get(model)
        if baseline_filename:
            baseline_path = os.path.join('model_outputs', dataset_name, 'baselines', baseline_filename)
            if os.path.exists(baseline_path):
                df_bl = pd.read_csv(baseline_path)
                baseline_metrics = {
                    'Raw Logits': 'raw_confidence',
                    'P(True)': 'p_true',
                }
                # Use baseline is_correct if row counts match, else fall back to main df
                bl_y_true = df_bl['is_correct'].values if 'is_correct' in df_bl.columns else None
                for bl_name, bl_col in baseline_metrics.items():
                    if bl_col in df_bl.columns:
                        bl_valid = df_bl[bl_col].notna()
                        bl_scores = df_bl.loc[bl_valid, bl_col].astype(float).values
                        if len(bl_scores) == 0:
                            continue
                        if bl_y_true is not None:
                            bl_yt = bl_y_true[bl_valid]
                        else:
                            bl_yt = y_true_soft[:len(bl_scores)]
                        bl_yt_binary = (bl_yt >= 0.5).astype(int)
                        ece = compute_ece(bl_yt, bl_scores)
                        disc, p_val = compute_discriminability(bl_yt_binary, bl_scores)
                        results.append({
                            'Dataset': dataset_name,
                            'Model': model,
                            'Metric': bl_name,
                            'ECE': ece,
                            'Discriminability': disc,
                            'P-Value': p_val
                        })

    return pd.DataFrame(results)

def format_significance(score, p_val):
    if pd.isna(score):
        return "NaN"
    
    star = ""
    if p_val < 0.001:
        star = "***"
    elif p_val < 0.01:
        star = "**"
    elif p_val < 0.05:
        star = "*"
        
    return f"{score:.3f}{star}"

def generate_summary_table(all_results, output_dir):
    all_results.to_csv(os.path.join(output_dir, 'ece_discriminability_raw.csv'), index=False)
    
    all_results['Discriminability_Formatted'] = all_results.apply(
        lambda row: format_significance(row['Discriminability'], row['P-Value']), axis=1
    )
    
    # Create a formatted table string or CSV
    
    pivot_disc = all_results.pivot_table(
        index='Metric', 
        columns=['Dataset', 'Model'], 
        values='Discriminability_Formatted',
        aggfunc='first' # Since values are strings now
    )
    
    pivot_ece = all_results.pivot_table(index='Metric', columns=['Dataset', 'Model'], values='ECE')
    model_order = ['llava-v1.5-7b', 'qwen2.5-vl-7b-instruct', 'gpt-4o-2024-05-13']
    
    desired_dataset_order = ['AOKVQA', 'VizWiz', 'MMMU-Pro']
    datasets = [ds for ds in desired_dataset_order if ds in set(all_results['Dataset'])]
    remaining = sorted(list(set(all_results['Dataset']) - set(datasets)))
    datasets.extend(remaining)
    # Calculate averages per dataset (across models)
    avg_disc = all_results.groupby(['Dataset', 'Metric'])['Discriminability'].mean().unstack(level=0)
    avg_ece = all_results.groupby(['Dataset', 'Metric'])['ECE'].mean().unstack(level=0)
    
    # Add Avg columns to pivot tables
    # pivot_disc has strings, so we format the average as string
    for ds in datasets:
        if ds in avg_disc.columns:
            # Create a Series for the new column
            avg_col_disc = avg_disc[ds].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
            # Add to pivot_disc with MultiIndex key
            pivot_disc[(ds, 'Avg')] = avg_col_disc
            
            # Add to pivot_ece (numeric)
            pivot_ece[(ds, 'Avg')] = avg_ece[ds]

    # Calculate Overall Average across all datasets
    overall_avg_disc = avg_disc.mean(axis=1)
    overall_avg_ece = avg_ece.mean(axis=1)

    # Add Overall Avg column
    pivot_disc[('Overall', 'Avg')] = overall_avg_disc.apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    pivot_ece[('Overall', 'Avg')] = overall_avg_ece

    new_columns = []
    for ds in datasets:
        for model in model_order:
            if (ds, model) in pivot_disc.columns:
                new_columns.append((ds, model))
        # Add Avg column after models
        if (ds, 'Avg') in pivot_disc.columns:
            new_columns.append((ds, 'Avg'))
            
    # Add Overall Avg at the very end
    if ('Overall', 'Avg') in pivot_disc.columns:
        new_columns.append(('Overall', 'Avg'))
                
    pivot_disc = pivot_disc.reindex(columns=new_columns)
    pivot_ece = pivot_ece.reindex(columns=new_columns)
    
    # Apply final formatting to ECE table (including Avg columns)
    # 3 decimal places as requested
    pivot_ece = pivot_ece.applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    
    # Reorder rows to match reference
    metric_order = [
        'Simulatability', 'Informativeness', 'Plausibility',
        'Visual Fidelity', 'Contrastiveness',
        'Avg(VF, Contr.)', 'Prod(VF, Contr.)', 'Min(VF, Contr.)',
        'Raw Logits', 'P(True)',
    ]
    
    pivot_disc = pivot_disc.reindex(metric_order)
    pivot_ece = pivot_ece.reindex(metric_order)
    
    print("\nDiscriminability Table:")
    print(pivot_disc)
    print("\nECE Table:")
    print(pivot_ece)
    
    pivot_disc.to_csv(os.path.join(output_dir, 'discriminability_pivot.csv'))
    pivot_ece.to_csv(os.path.join(output_dir, 'ece_pivot.csv'))

if __name__ == '__main__':
    datasets = ['AOKVQA', 'VizWiz', 'MMMU-Pro']
    models = ['llava-v1.5-7b', 'qwen2.5-vl-7b-instruct', 'gpt-4o-2024-05-13']
    output_dir = 'ece_analysis_results'
    
    all_results_list = []
    
    for dataset in datasets:
        df_res = analyze_dataset(dataset, models, output_dir)
        all_results_list.append(df_res)
        
    final_df = pd.concat(all_results_list, ignore_index=True)
    generate_summary_table(final_df, output_dir)
