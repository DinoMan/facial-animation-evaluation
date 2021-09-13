# Facial Animation Evaluation
Script and library for evaluation of facial animation videos. Do you find yourself wanting to evaluate you videos of talking heads but don't know what metrics to use or how to implement them. Well then this repo is for you!

# Installation


## Evaluating Videos
You can use the **evaluate.py** script to calculate metrics for your videos. 
```
python evaluate.py -i <video folder> -r <reference video folder> -a <annotations file>
```
#### Arguments
Argument | Short-hand | Type | Description
-------- | :--------: | ---- | --------
--input | -i | **required** | Folder containing generated videos to be evaluated |
--reference| -r | optional | Folder containing reference videos for full reference metrics |
--annotation | -a | optional | File containing annotations for the videos (regex or files) |
--lipreading | -l | optional | Path to lipreading model |
--emotion_net | -e | optional | Path to emotion classification model (will use emonet by default)|
--num_emotions |  | optional | Number of emotions supported by the emotion classifier|
--emotion_aggregation |  | optional | How to aggregate the emotions for all frames (voting or average)|
--ignore_emotions |  | optional | Which emotions should be ignored |
--gpu | -g | optional | Should the available GPUs be used (makes script much faster)|
--full_report | | optional | The file path to store the full report (containing per file metrics)|

## Using the library
In some cases you may want to calculate evaluation metrics while you train a model. 
In this case you can use the **MouthEvaluator** and **EmotionEvaluator** classes or the **calculate_full_reference_metrics**, and **calculate_no_reference_metrics** methods.     


### Acknowledgements
- The default emotion detector used is from [Toisoul et.al (2021)](https://www.nature.com/articles/s42256-020-00280-0) [[code]](https://github.com/face-analysis/emonet). Please be aware that this detector can not be used for commercial purposes.
- The default lipreader for LRW is based on [Petridis et.al. (2018)](https://ieeexplore.ieee.org/document/8461326) [[code]](https://github.com/mpc001/end-to-end-lipreading)
