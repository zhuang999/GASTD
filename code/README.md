# GASTD: Graph-Augmented Spatial-Temporal Distillation for Efficient Next POI Recommendation
![image](fig/architecture.pdf)
# Requirements
```
pip install -r requirements.txt
```

# Data Preparation

**1. place [flashback_data.zip](https://drive.google.com/file/d/1QXdpp0_QesJo7NZdhvoafg4MlpI_Bx-O/view?usp=sharing) into /data/ and unzip the file as following:**

/data/checkins-gowalla.txt

/data/checkins-4sq.txt

<!-- https://drive.google.com/file/d/1ST6GQidWVlR6yQle38MfPUSUc29t9xIT/view?usp=sharing -->

**2. place [Graphs.zip](https://drive.google.com/file/d/1KC361Gq-K-0Aw7xu5pyl51YOgMK9JtMb/view?usp=sharing) into Graph_Flashback/KGE/ and unzip the file as following :**

/KGE/Graphs/gowalla_scheme2_transe_loc_temporal_100.pkl

# Model Training

```
python train.py
```
