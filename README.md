# SplatfactoTrainer

This script trains the splatfacto model with as little overhead as possible.
This means the viewer and webserver aren't run and much less is printed to
the console. The model is trained directly from the input images, including
running COLMAP and we save directly to a .splat file. The inputs to this
script are on lines 21-36.

The dependencies of the script are nerfstudio, gsplat, and PyTorch.