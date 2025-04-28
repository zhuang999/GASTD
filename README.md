# GASTD: Graph-Augmented Spatial-Temporal Distillation for Efficient Next POI Recommendation
![architecture](fig/architecture.png)
# Requirements
```
pip install -r requirements.txt
```

# Data Preparation

**1. place [flashback_data.zip](https://drive.google.com/file/d/1QXdpp0_QesJo7NZdhvoafg4MlpI_Bx-O/view?usp=sharing) into /data/ and unzip the file as following:**

/data/checkins-gowalla.txt

/data/checkins-4sq.txt

<!-- https://drive.google.com/file/d/1ST6GQidWVlR6yQle38MfPUSUc29t9xIT/view?usp=sharing -->

**2. place [Graphs.zip](https://drive.google.com/file/d/1KC361Gq-K-0Aw7xu5pyl51YOgMK9JtMb/view?usp=sharing) into KGE/ and unzip the file as following :**

/KGE/gowalla_scheme2_transe_loc_temporal_100.pkl

# Model Training

```
python train.py
```

# Motivation

The findings from the data visualization analysis are as follows: (a) diverse data augmentation improved the model's predictive performance, (b) the student model derived from standard knowledge distillation lacks confidence, (c) the student model's confidence is positively correlated with the teacher model's confidence, and (d) the student model's prediction errors are concentrated in areas where the teacher model has low confidence. These insights motivate us to enhance the student's confidence to improve its lightweight prediction accuracy. 
![motivation](fig/intro.png)

# Trajectory Completion Module (TCM)

To enhance the spatial-temporal information and confidence of the student model, we mined the trajectory to identify potentially missing check-ins.
![TCM](fig/bfs.png)

# Efficiency

We leveraged a reliable multi-teacher knowledge distillation framework, which not only improved the prediction accuracy of the student model but also retained its efficiency advantage.
![efficiency](fig/inference_time.png)