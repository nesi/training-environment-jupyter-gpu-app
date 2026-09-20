#!/bin/bash -e
#SBATCH --job-name      watch-me
#SBATCH --account       nesi99991
#SBATCH --time          00:15:00
#SBATCH --cpus-per-task 2
#SBATCH --mem           4GB
#SBATCH --gpus-per-node l4:1
#SBATCH --output        watch-me-%j.out

# A job that runs long enough to go and look at it.
#
# Submit it, then watch it live:
#
#   sbatch watch-me.sl
#   squeue --me                 # wait until ST is R, and note the JOBID
#   svisit <JOBID>              # open a terminal on the node running the job
#   nvtop                       # watch it
#
# In nvtop, press q to quit, then 'exit' to leave the node.
#
# What to look for, in this order:
#
#   1. The GPU% trace rising and falling every 30 seconds. That is the job's
#      own rhythm - 20 seconds of work, 10 seconds of waiting.
#   2. The MEM trace, which goes up once and then does not move. Memory is
#      held, not consumed.
#   3. The process list at the bottom. Your python3 should be there. If the
#      GPU is busy and your process is NOT listed, you are watching somebody
#      else's job.

python3 sawtooth.py --minutes 8
