# # Experiment 1
# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --wrap="python -u ../src/experiment1.py --lm glove"
# # sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --gres=gpu:a100:1 --wrap="python -u ../src/experiment1.py --lm mbert"
# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --gres=gpu:a100:1 --wrap="python -u ../src/experiment1.py --lm roberta"

# # Experiment 2
# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --wrap="python -u experiment2.py --lm glove --size 2000"
# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --gres=gpu:a100:1 --wrap="python -u experiment2.py --lm mbert --size 2000"
# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --gres=gpu:a100:1 --wrap="python -u experiment2.py --lm roberta --size 2000"

# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --wrap="python -u experiment2.py --lm glove --size 4000"
# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --gres=gpu:a100:1 --wrap="python -u experiment2.py --lm mbert --size 4000"
# sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --gres=gpu:a100:1 --wrap="python -u experiment2.py --lm roberta --size 4000"

# # Old job for reference
# sbatch -J codi-eval -p compute --mem 50000 -t 2-0 --gres=gpu:a100:1 --cpus-per-task=8 --wrap="python3 -u WCTest.py"

sbatch -J linearity -p compute --mem 80000 -t 2-0 --cpus-per-task=4 --wrap="python -u ../src/BATS.py"
