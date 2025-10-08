import matplotlib.pyplot as plt
import os 

def visualize_components(trend, season, step, save_dir="OUTPUT/plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(trend[0,:,1].detach().cpu().numpy(), label="Trend")
    plt.plot(season[0,:,1].detach().cpu().numpy(), label="Season")
    plt.legend()
    plt.title(f"Step {step}")
    plt.savefig(f"{save_dir}/components_step_{step}.png")
    plt.close()
