<p align="center">
  <img src="docs/banner.png?raw=true" width="750" height="375">
</p>
**blearn** is a tool that allows you to generate an image-based dataset in Blender. Together with a mesh and texture obtained from photogrammetry, realistic synthetic datasets can be generated. The script is executed with Blender built-in python interpreter, which has the advantage that `bpy` is already loaded correctly

### Requirements
Install all requirements with:
```bash
pip3 install -r requirements.txt
```

### Example
```bash
blender -b <path_to_blend_file>.blend --python blearn.py
```

### Running tests
```bash
python -m pytest
```

### Offload to GPU with Nvidia prime on-demand
When using Nvidia prime with the on-demand setting, the use of the GPU must be force using the following env definitions:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia blender <path_to_blend_file>.blend --python blearn.py
```
