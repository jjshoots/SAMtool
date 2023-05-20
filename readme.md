# SAMTool - Semantic Segmentation Dataset Creation powered by Segment Anything Model from Meta

<p align="center">
    <img src="demo.gif" width="800"/>
</p>

## Installation

`pip3 install git+https://github.com/jjshoots/SAMtool`

## Usage

1. `samtool --imagedir <images directory> --labeldir <labels directory> --annotations <annotations.yaml file>`
2. Go to `127.0.0.1:7860`

### Defining labels

The labels must be defined as a `yaml` file. Example contents of the file:
```yaml
__ignore__: 0
cat: 1
dog: 2
other: 3
```
