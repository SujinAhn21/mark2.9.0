# mark2.9.0


## Key Features of mark2.5.0 ~ mark2.9.0  

### 1. Dual Label Input Structure 
* These models utilize two types of labels per audio sample:  
  - Hard Label: Human-annotated binary class labels (e.g., noise vs. no noise)  
  - Soft Label: Probability distributions predicted by a pre-trained teacher model (knowledge distillation)  
* By combining both labels during training, the model learns not only to classify correctly but also to mimic the reasoning of the teacher.  


### 2. Combined Dual-Loss Strategy  
* The DistillationLoss class combines two types of losses:  
  - Hard Loss: CrossEntropyLoss — supervised classification based on ground truth  
  - Soft Loss: KLDivLoss — mimics the probabilistic distribution of the teacher model  

* 전체 손실 계산 공식:  
  'total_loss = α * SoftLoss + (1 - α) * HardLoss'  

  * α (alpha): Soft loss의 가중치 (기본값 0.7)  
  * T (Temperature): 확률 분포 스무딩 조절 파라미터 (기본값 4.0)  


### 3. Dynamic Feature Alignment Architecture)  

- Audio input is segmented, encoded, and classified per segment before final integration.  
- 전체 모델 구조:  
  'Audio -> Encoder -> Flatten -> Head -> Prediction (logits)'  


### Hybrid Early Stopping and LR Scheduler  

* To prevent overfitting and improve generalization, the following techniques are applied:  
  - EarlyStopping: stops training when validation performance stagnates  
  - ReduceLROnPlateau: reduces learning rate when validation loss plateaus  


### Model Checkpointing and Visualization  

* The best-performing model’s encoder and head are automatically saved.  
* Training and validation loss curves are exported as `.png` files for further analysis.  


## Difference from mark2.1.x ~ 2.4.0  

* Starting from mark2.5.0, a  'hybrid learning strategy' was introduced by combining 'CrossEntropyLoss (hard labels)' with 'KLDivLoss (soft labels)' for joint supervision and knowledge distillation.
* This logic has been implemented in versions mark2.5.0 through mark2.9.0.   


## License
- This project is licensed under the PolyForm Noncommercial License 1.0.0.  
- Commercial use is not permitted.
- See the [LICENSE](./LICENSE) file for details.


