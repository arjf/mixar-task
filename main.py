# %%
import logging as log
from helpers import *
from MeshProcessor import MeshProcessor

setup_logging()
lg = log.getLogger(__name__)

# %%
target = "talwar"
original = load_mesh(samplepath(target))
obj = MeshProcessor(load_mesh(samplepath(target)), base_bins=1024)

obj.process(normalization="unit_sphere", quantization="adaptive")

normalized = obj.normalized_mesh
processed = obj.quantized_mesh

translation_amount = 2.2
processed.apply_translation([translation_amount, 0, 0])
lg.info(f"Moving processed mesh by {translation_amount:.2f} units for viewing.")

scene = tri.Scene()
scene.add_geometry(normalized)
scene.add_geometry(processed)
scene.show()


#%%