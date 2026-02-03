#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --output=jupyter_notebook_%j.txt
#SBATCH --error=jupyter_notebook_%j.err

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="discovery8"

port=`echo $(( 8888 ))`

# print tunneling instructions jupyter-log
echo -e "
# Note: below 8888 is used to signify the port.
#       However, it may be another number if 8888 is in use.
#       Check jupyter_notebook_%j.err to find the port.

# Command to create SSH tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.dartmouth.edu

# Use a browser on your local machine to go to:
http://localhost:${port}/
"

module load python
jupyter-lab --no-browser --ip=${node} --port=${port} --ResourceUseDisplay.track_cpu_percent=True

# keep it alive
sleep 36000
