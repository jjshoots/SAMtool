# SAMTool - Semantic Segmentation Dataset Creation powered by Segment Anything Model from Meta

<p align="center">
    <img src="demo.gif" width="800"/>
</p>

## Installation

`pip3 install git+https://github.com/jjshoots/SAMtool`

## Usage

1. `samtool --imagedir <images directory> --labeldir <labels directory> --annotations <annotations.yaml file>`
2. Go to `127.0.0.1:7860`

#### Defining labels

The labels must be defined as a `yaml` file. Example contents of the file:
```yaml
__ignore__: 0
cat: 1
dog: 2
other: 3
```

## Labels Storage

All labels are stored as a series of jpg files on the disk.
To operate on labels, we provide several helper functions:

### To retrieve labels

```python
import os
from samtool import retrieve_label

all_labels = yaml.safe_load(open(annotations_path))
num_labels = len(all_labels)

for image_filename in os.listdir("./your_image_dir"):
    npy_label = retrieve_label(label_dir="./your_label_dir", image_filename=image_filename, num_labels=num_labels)
```

### To check if a label exists

```python
import os
from samtool import label_exists

all_labels = yaml.safe_load(open(annotations_path))
num_labels = len(all_labels)

for image_filename in os.listdir("./your_image_dir"):
    has_label = label_exists(label_dir="./your_label_dir", image_filename=image_filename, num_labels=num_labels)
```
