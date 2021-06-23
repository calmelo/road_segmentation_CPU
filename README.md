# Road segmentation just using CPU

This repo contains the following file folder:
- data: some experimental data.
- model: openvino's pretrained model. (https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/road-segmentation-adas-0001)
- results: the experimental results of processing the experimental data in the data file folder.
- scripts: the py file. Using the road_segmentation_images.py to process the image folder, and using the road_segmentation_video.py to process the video.

# Setup

### System requirement

- Ubuntu18.04
- Python 3.6 or higher
- OpenVINO 2021.3 ([download](https://software.intel.com/en-us/openvino-toolkit/choose-download))
- numpy (`pip3 install numpy`)
- No GPU requirement

### Download

git clone https://github.com/calmelo/road_segmentation_CPU

# Run

. /opt/intel/openvino/bin/setupvars.sh

cd road_segmentation_CPU/scripts

python3 road_segmentation_images.py/road_segmentation_video.py
