# Gait CAL

## Train

1. Config at ```cfg.py```
2. Create Gait CAL Phase 2 Dataset (Autoencoders Features and GEI)  
```create_casia_b_phase_2_dataset.py```
3. Train Gait CAL
```train_phase_2_gait_cal_on_casia_b.py```
4. Set trained model at ```create_gallery_nm_1_4.py``` and run it to create database
5. Run ``eval.py`` to export result.