# %%
import logging as log
import json
from helpers import *
from visualize import *
from MeshProcessor import MeshProcessor
from collections import defaultdict

setup_logging()
lg = log.getLogger(__name__)

# %%
# Task 1 - Load and Inspect the Mesh
target = "talwar" # This will be target sample used for all visualizations going foward
original = load_mesh(samplepath(target))
obj = MeshProcessor(load_mesh(samplepath(target)), base_bins=1024)
print(f"{target} Mesh Stats: {obj.get_stats(obj.vertices)}")
# obj.show()

# %%
# Task 2: Normalize and Quantize the Mesh
obj1 = MeshProcessor(load_mesh(samplepath(target)), base_bins=1024)
obj2 = MeshProcessor(load_mesh(samplepath(target)), base_bins=1024)
obj1.process(normalization="minmax", quantization="uniform")
obj2.process(normalization="unit_sphere", quantization="uniform")
print(obj1.get_stats(obj1.vertices))
print(obj2.get_stats(obj2.vertices))

sc1 = side_by_side_visualization(obj1.normalized_mesh, obj1.quantized_mesh, titles = ("Minmax - Normalized mesh", "Uniformly quantized mesh"), show=False)
sc2 = side_by_side_visualization(obj2.normalized_mesh, obj2.quantized_mesh, titles = ("Unit sphere - Normalized mesh", "Uniformly quantized mesh"), show=False)

sc1.export(f"results/{target}_MinMaxNorm-UniformQuant.obj")
sc2.export(f"results/{target}_UnitSphereNorm-UniformQuant.obj")

MeshProcessor.export(obj1.normalized_mesh, target, nick="normalized_MinMaxNorm-UniformQuant", path="./results")
MeshProcessor.export(obj1.quantized_mesh, target, nick="quantized_MinMaxNorm-UniformQuant", path="./results")
MeshProcessor.export(obj2.normalized_mesh, target, nick="normalized_UnitSphereNorm-UniformQuant", path="./results")
MeshProcessor.export(obj2.quantized_mesh, target, nick="quantized_UnitSphereNorm-UniformQuant", path="./results")

# %%
# Task 3: Dequantize, Denormalize, and Measure Error

def process_all():
    sample_dir = "./data/8samples/"
    samples = [f.split(".")[0] for f in os.listdir(sample_dir) if f.endswith(".obj")]
    lg.debug(f"Processing samples: {samples}")
    results = defaultdict(defaultdict) #dict.fromkeys(samples, value=list())
    norm = ["unit_sphere", "minmax"]
    quant = ["adaptive", "uniform"]
    
    res_dir = './results'
    if not os.path.exists(res_dir): os.mkdir(res_dir)
    
    for sample in samples:
        lg.info(f"Processing Sample: {sample}")
        path = f"./{res_dir}/{sample}"
        if not os.path.exists(path): os.mkdir(path)
        sample_mesh = load_mesh(samplepath(sample))
        for n in norm:
            for q in quant:
                combo_id = f"{n}_Norm-{q}_Quant"
                
                obj = MeshProcessor(sample_mesh.copy(), base_bins=1024)
                obj.process(normalization=n, quantization=q)
                
                original_vertices = obj.mesh.vertices
                denormalized_mesh = obj.denormalize()
                denormalized_vertices = denormalized_mesh.vertices
                
                pt = f"./{res_dir}/{sample}/models/"
                if not os.path.exists(pt): os.mkdir(pt)
                obj.export(obj.normalized_mesh, sample, nick=f"normalized-{combo_id}", path=pt, appened_time=False)
                obj.export(denormalized_mesh, sample, nick=f"denormalized-{combo_id}", path=pt, appened_time=False)
                lg.debug(f"Exported meshes to {path}")
                
                # get metrics
                metrics_data = {
                    "overall_mae": mae(original_vertices, denormalized_vertices, axis_wise=False),
                    "overall_mse": mse(original_vertices, denormalized_vertices, axis_wise=False),
                    "overall_rmse": rmse(original_vertices, denormalized_vertices, axis_wise=False),
                    "axis_mae": mae(original_vertices, denormalized_vertices, axis_wise=True),
                    "axis_mse": mse(original_vertices, denormalized_vertices, axis_wise=True),
                    "axis_rmse": rmse(original_vertices, denormalized_vertices, axis_wise=True)
                }
                
                results[sample][combo_id] = metrics_data
                
                # viz
                plot_prefix = f"{sample}_{combo_id}"
                figures = plot_error_per_axis(original_vertices,denormalized_vertices, pfx=plot_prefix, show=False)
                bar_chart_fig = figures[0]
                hist_fig = figures[1]
                bar_chart_path = os.path.join(path, f"{plot_prefix}_summary_bars.png")
                hist_path = os.path.join(path, f"{plot_prefix}_error_dist.png")
                bar_chart_fig.savefig(bar_chart_path)
                hist_fig.savefig(hist_path)
                plt.close(bar_chart_fig)
                plt.close(hist_fig)
                lg.debug(f"Saved plots to {path}")
        
    metrics_json_path = os.path.join(res_dir, "all_metrics_summary.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    lg.info(f"All processing complete. Metrics saved to {metrics_json_path}")
    return results
                
# We are saving the obj files for the deconstructed and normalized mesh for each object with each combination of normalization and quantization 
results = process_all()

original = MeshProcessor(load_mesh(samplepath(target)))
original.process(normalization="unit_sphere", quantization="adaptive")
denormalized_mesh = original.denormalize()

# plot reconstruction error and metrics
figs = plot_error_per_axis(original.vertices, denormalized_mesh.vertices, pfx="quant_", show=True)

