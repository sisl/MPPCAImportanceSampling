# packages
import matplotlib.pyplot as plt

# file imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from problems import F16GCAS, Branches, Oscillator

# latex rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for all text rendering
    "font.family": "serif",  # Use serif font (default LaTeX font)
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}"  # Load extra packages
})
# Create the figure
fig, axs = plt.subplots(3,5,figsize=(15,9),sharey='row')

#*******************************************************************************
# Branches plot
#*******************************************************************************
problem = Branches()
_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_mc_dataset()
problem.plot(axs[0,0],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("CE-MPPCA",'./results/')
problem.plot(axs[0,1],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("SIS-MPPCA",'./results/')
problem.plot(axs[0,2],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("CE-GMM",'./results/')
problem.plot(axs[0,3],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("SIS-GMM",'./results/')
problem.plot(axs[0,4],failure_trajectories, non_failure_trajectories)

#*******************************************************************************
# Oscillator plot
#*******************************************************************************
problem = Oscillator(d=100)
_, _, failure_trajectories, non_failure_trajectories, _  = problem.load_mc_dataset()
problem.plot(axs[1,0],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("CE-MPPCA",'./results/')
problem.plot(axs[1,1],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("SIS-MPPCA",'./results/')
problem.plot(axs[1,2],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("CE-GMM",'./results/')
problem.plot(axs[1,3],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("SIS-GMM",'./results/')
problem.plot(axs[1,4],failure_trajectories, non_failure_trajectories)

#*******************************************************************************
# F-16 GCAS plot
#*******************************************************************************
problem = F16GCAS()
_, _, failure_trajectories, non_failure_trajectories, _  = problem.load_mc_dataset()
problem.plot(axs[2,0],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("CE-MPPCA",'./results/')
problem.plot(axs[2,1],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("SIS-MPPCA",'./results/')
problem.plot(axs[2,2],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("CE-GMM",'./results/')
problem.plot(axs[2,3],failure_trajectories, non_failure_trajectories)

_, _, failure_trajectories, non_failure_trajectories, _ = problem.load_results("SIS-GMM",'./results/')
problem.plot(axs[2,4],failure_trajectories, non_failure_trajectories)

#*******************************************************************************
# plot styling
#*******************************************************************************
[[axs[i,j].set_ylabel("") for j in range(1,axs.shape[1])] for i in range(axs.shape[0])] # remove the ylabel because of shared y-axis

fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.06, hspace=0.23)

# method headings
for i,h in enumerate([r"\textbf{MC}", r"\textbf{CE-MPPCA}", r"\textbf{SIS-MPPCA}", r"\textbf{CE-GMM}", r"\textbf{SIS-GMM}"]):
    pos = axs[0,i].get_position()
    fig.text(pos.x0 + pos.width / 2, pos.y0+pos.height*1.1, h, ha='center', fontsize=12, fontweight='bold')
    
# Problem Headings
for i,h in enumerate([r"\textbf{Branches ($\mathbf{d=40}$)}", r"\textbf{Oscillator ($\mathbf{d=100}$)}", r"\textbf{F-16 GCAS}"]):
    pos = axs[i,0].get_position()
    fig.text(pos.x0 -0.32*pos.width, pos.y0+pos.height/2, h, va='center', ha='center', fontsize=12, fontweight='bold', rotation=90)

plt.savefig("results.pdf")

print("Saved results.")