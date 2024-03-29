# A Mini-app Example for Scientific Visualization Using OSPRay

Thanks Will Usher for sharing the [OSPRay viewer](https://github.com/Twinklebear/mini-osp-vis).
This flow map viewer is built mini-osp-viewer with adding neural network inference, pathline rendering and FTLE interpolation. 

Dependencies:

- OSPRay 2.0
- TBB
- SDL2
- GLM (Use the latest https://github.com/g-truc/glm or release 0.9.9.8 or higher)
- VTK (optional) for computing explicit triangle isosurfaces for testing
- OpenVisus (optional) for loading IDX volumes

Use the provided "fetch_scivis.py" script to fetch a dataset and its
JSON metadata from [OpenScivisDatasets](https://klacansky.com/open-scivis-datasets/).
The script requires the [requests](https://requests.readthedocs.io/en/master/) library.

