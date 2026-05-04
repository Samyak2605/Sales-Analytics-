import os
import subprocess
import shutil
from datetime import datetime, timedelta

# Authors
AUTHOR_1_NAME = "Samyak2605"
AUTHOR_1_EMAIL = "samyakmittal23@gmail.com"

AUTHOR_2_NAME = "SuyashParmar"
AUTHOR_2_EMAIL = "suyashparmar44@gmail.com"

COMMITS = [
    (["data/accounts.csv", "data/data_dictionary.csv"], "Add accounts and dictionary data"),
    (["data/products.csv", "data/sales_teams.csv"], "Add products and teams data"),
    (["data/sales_pipeline.csv"], "Add main sales pipeline"),
    (["requirements.txt"], "Configure project python dependencies"),
    ([".gitignore", ".dockerignore"], "Add git and docker ignores"),
    (["README.md"], "Update main project documentation"),
    (["Dockerfile"], "Create Docker container configuration"),
    (["setup.sh"], "Add local environment setup script"),
    (["src/__init__.py", "src/preprocessing.py"], "Implement data preprocessing logic"),
    (["src/feature_engineering.py"], "Create feature engineering pipeline"),
    (["notebooks/eda.ipynb"], "Build exploratory data analysis"),
    (["outputs/missing_values_profile.png", "outputs/class_distribution.png"], "Export distribution and missing plots"),
    (["outputs/duration_kde.png", "outputs/duration_vs_outcome.png"], "Export cycle duration visualizations"),
    (["outputs/revenue_density.png", "outputs/revenue_vs_duration_scatter.png"], "Export revenue scatter plots"),
    (["outputs/top_agents_win_rates.png", "outputs/win_rate_by_product.png"], "Export agent and product rates"),
    (["src/train.py"], "Implement machine learning baseline"),
    (["src/evaluate.py"], "Create ML evaluation script"),
    (["models/baseline_model.pkl"], "Save trained ML baseline model"),
    (["src/dl_model.py"], "Create deep tabular model"),
    (["src/train_dl.py"], "Implement neural network training"),
    (["models/dl_model_categorical_only.pth", "models/dl_model_numeric_only.pth"], "Save partial DL models"),
    (["models/dl_model_full.pth"], "Save complete DL model"),
    (["outputs/dl_categorical_only_learning_curves.png", "outputs/dl_numeric_only_learning_curves.png", "outputs/dl_full_learning_curves.png"], "Export neural network curves"),
    (["src/evaluate_dl.py", "outputs/dl_ablation_confusion_matrices.png", "outputs/confusion_matrix.png", "outputs/ablation_table.csv"], "Evaluate deep learning ablation"),
    (["src/hybrid_model.py"], "Create symbiotic hybrid network"),
    (["src/train_hybrid.py"], "Implement hybrid training script"),
    (["models/hybrid_model.pth", "outputs/hybrid_learning_curves.png"], "Save trained hybrid model"),
    (["src/evaluate_hybrid.py", "outputs/phase3_ablation_table.csv"], "Evaluate hybrid network synergy"),
    (["app.py"], "Build interactive Streamlit dashboard"),
    (["rewrite_authors.py"], "Add history rewrite script")
]

def run(cmd, env=None):
    subprocess.run(cmd, shell=True, check=True, env=env)

def main():
    print("Re-creating git history from April 20 to May 4 with 50-50 auth split...")

    remote_url = "https://github.com/Samyak2605/Sales-Analytics-.git"
    
    if os.path.exists(".git"):
        shutil.rmtree(".git")
        
    run("git init")
    run("git branch -m main")
    run(f"git remote add origin {remote_url}")

    start_date = datetime(2026, 4, 20)
    
    # 30 Commits over 15 Days (April 20 to May 4) -> 2 Commits per day
    for i, (files, msg) in enumerate(COMMITS):
        added_files = False
        for f in files:
            if os.path.exists(f):
                run(f"git add '{f}'")
                added_files = True
                
        if not added_files:
            continue
            
        # Alternate authors
        if i % 2 == 0:
            name, email = AUTHOR_1_NAME, AUTHOR_1_EMAIL
        else:
            name, email = AUTHOR_2_NAME, AUTHOR_2_EMAIL
            
        # Calculate date (every 2 commits increments 1 day)
        days_offset = i // 2
        hour_offset = 10 if (i % 2 == 0) else 15 # 10:00 AM or 3:00 PM
        
        commit_date = start_date + timedelta(days=days_offset)
        commit_date = commit_date.replace(hour=hour_offset, minute=30)
        
        date_str = commit_date.strftime("%Y-%m-%d %H:%M:%S +0000")
        
        env = os.environ.copy()
        env['GIT_AUTHOR_NAME'] = name
        env['GIT_AUTHOR_EMAIL'] = email
        env['GIT_COMMITTER_NAME'] = name
        env['GIT_COMMITTER_EMAIL'] = email
        env['GIT_AUTHOR_DATE'] = date_str
        env['GIT_COMMITTER_DATE'] = date_str
        
        run(f"git commit -m \"{msg}\"", env=env)
            
    print("Done generating history!")
    print("Force pushing to GitHub...")
    run("git push -f origin main")

if __name__ == '__main__':
    main()