import subprocess
import time
import os
import json
import sys
from datetime import datetime

TEST_SET = r"E:\Multi_Agent_api\dataset\test_set_1000.json"

def run_cmd(cmd_list):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] RUNNING: " + " ".join(cmd_list))
    subprocess.run(cmd_list, check=True)

def main():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LLM_TIMEOUT", "180")
    os.environ.setdefault("LLM_MAX_RETRIES", "0")
    base_cmd = [sys.executable, "ablations/run_experiment.py", "--input_file", TEST_SET]
    
    experiments = [

        # --- 1. Feature Ablations ---
        ["--exp_name", "feat_no_rwr", "--disable_rwr"],
        ["--exp_name", "feat_no_sim", "--disable_sim"],
        ["--exp_name", "feat_no_graph", "--disable_graph"],
        ["--exp_name", "feat_no_llm", "--disable_llm"],
        
        # --- 2. Routing Ablations ---
        ["--exp_name", "route_no_rules", "--disable_rules"],
        ["--exp_name", "route_no_fast_cot", "--disable_fast_cot"],
        ["--exp_name", "route_no_full_review", "--disable_review"],
        
        # --- 3. Hyperparameters (Uncertainty Windows) ---
        ["--exp_name", "hyper_window_narrow", "--uncertainty_low", "0.55", "--uncertainty_high", "0.60"],
        ["--exp_name", "hyper_window_wide", "--uncertainty_low", "0.45", "--uncertainty_high", "0.70"],
        
        # --- 4. Hyperparameters (RWR Restart) ---  
        ["--exp_name", "hyper_rwr_0.1", "--rwr_restart_prob", "0.1"],
        ["--exp_name", "hyper_rwr_0.7", "--rwr_restart_prob", "0.7"],
        
        # --- 5. Hyperparameters (Hard Filter) ---
        ["--exp_name", "hyper_filter_1000", "--rwr_hard_filter_rank", "1000"],
        ["--exp_name", "hyper_filter_2000", "--rwr_hard_filter_rank", "2000"],

        # --- 6. Prompting Strategies ---
        ["--exp_name", "prompt_zeroshot_direct", "--prompt_strategy", "zeroshot_direct"],
        ["--exp_name", "prompt_zeroshot_cot", "--prompt_strategy", "zeroshot_cot"],
        ["--exp_name", "prompt_evidence_first", "--prompt_strategy", "evidence_first"],
        ["--exp_name", "prompt_json_strict", "--prompt_strategy", "json_strict"],
        ["--exp_name", "prompt_calibrated_cot", "--prompt_strategy", "calibrated_cot"],
    ]
    expected_exp_names = {
        args[args.index("--exp_name") + 1]
        for args in experiments
        if "--exp_name" in args
    }
    
    os.makedirs("ablations/results_gpt", exist_ok=True)
    timing_file = "ablations/results_gpt/experiment_timing_report.txt"
    
    with open(timing_file, 'w', encoding='utf-8') as tf:
        tf.write("=== Ablation Experiments Timing Report ===\n\n")
        tf.write(f"Test set: {TEST_SET}\n")
        tf.write("Metrics: ACC uses threshold score >= 0.5; AUC/AUPR use continuous predicted scores.\n\n")
        
    for args in experiments:
        exp_name = args[args.index("--exp_name") + 1] if "--exp_name" in args else "unknown"
        
        start_t = time.time()
        start_dt = datetime.now()
        
        status = "SUCCESS"
        try:
            run_cmd(base_cmd + args)
        except subprocess.CalledProcessError:
            status = "FAILED"
            
        end_t = time.time()
        end_dt = datetime.now()
        duration = end_t - start_t
        h, rem = divmod(duration, 3600)
        m, s = divmod(rem, 60)
        dur_str = f"{int(h):02d}:{int(m):02d}:{s:05.2f}"
        
        with open(timing_file, 'a', encoding='utf-8') as tf:
            tf.write(f"Experiment : {exp_name}\n")
            tf.write(f"Status     : {status}\n")
            tf.write(f"Start Time : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            tf.write(f"End Time   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            tf.write(f"Duration   : {dur_str} ({duration:.2f}s)\n")
            tf.write("-" * 50 + "\n")
        
    print(f"\nAll experiments finished. Timings saved to {timing_file}")
    
    # Generate quick summary
    res_dir = "ablations/results_gpt"
    summary = []
    if os.path.exists(res_dir):
        for f in os.listdir(res_dir):
            if f.endswith('.json'):
                try:
                    with open(os.path.join(res_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        metrics = data.get('metrics', {})
                        if metrics and metrics.get('exp_name') in expected_exp_names:
                            summary.append(metrics)
                except:
                    pass
    
    # Sort and display
    summary.sort(key=lambda x: x.get('exp_name', ''))
    
    summary_table = [
        f"\n{'Experiment Name':<25} | {'ACC':<6} | {'AUC':<6} | {'AUPR':<6} | {'Time(s)':<7} | {'Rev Rate':<5}",
        "-" * 69
    ]
    for s in summary:
        summary_table.append(
            f"{s.get('exp_name', 'N/A'):<25} | {s.get('accuracy',0):.4f} | {s.get('auc',0):.4f} | {s.get('aupr',0):.4f} | {s.get('time_seconds',0):.1f} | {s.get('review_rate',0):.1%}"
        )
        
    summary_out = "\n".join(summary_table)
    print(summary_out)
    
    # Also append the final summary table to the text file
    with open(timing_file, 'a', encoding='utf-8') as tf:
        tf.write("\n=== Final Performance Summary ===\n")
        tf.write(summary_out + "\n")

if __name__ == "__main__":
    main()
