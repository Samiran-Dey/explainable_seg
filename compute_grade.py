import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, top_k_accuracy_score, confusion_matrix 
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt

save_path='Code/checkpoints/ricord_10/'
grade_path = 'Code/checkpoints/ricord_gt/grade_predictions.xlsx'
grade_df = pd.read_excel(grade_path)

c_id=list(grade_df['Case_ID'])
s_id=list(grade_df['Series_ID'])
sex=list(grade_df['Sex'])
age=list(grade_df['Age'])
gt=list(grade_df['Ground_truth'])
pred=list(grade_df['Prediction'])

def remarks(grade):
  if grade==0:
    return "no edema"
  if grade<8:
    return "mild"
  elif grade<=15:
    return "moderate"
  else:
    return "severe"
    
def remarks_no(grade):
  if grade==0:
    return 0
  if grade<8:
    return 1
  elif grade<=15:
    return 2
  else:
    return 3
    
gt_grade, pred_grade, gt_no, pred_no = [], [], [], []
for i in range(len(gt)):
  gt_grade.append(remarks(gt[i]))
  pred_grade.append(remarks(pred[i]))
  gt_no.append(remarks_no(gt[i]))
  pred_no.append(remarks_no(pred[i]))
  
pgrade_df = pd.DataFrame(list(zip(c_id, s_id, age, sex, gt, pred, gt_grade, pred_grade)), columns =['Case_ID', 'Series_ID', 'Age', 'Sex', 'Ground_truth', 'Prediction', 'Ground_truth_Remark', 'Prediction_Remark'])
file_name = save_path+'grade_predictions_gt.xlsx'
pgrade_df.to_excel(file_name)

kappa=cohen_kappa_score(gt, pred, weights='quadratic')
kappa_grade=cohen_kappa_score(gt_no, pred_no, weights='quadratic')
acc = accuracy_score(gt, pred)
acc_grade = accuracy_score(gt_no, pred_no)

msg = 'Result_score->\n kappa=%.7f\n accuracy=%.7f \nResult_grade->\n kappa=%.7f\n accuracy=%.7f'%(kappa, acc, kappa_grade, acc_grade)
print(msg)

log_name = os.path.join(save_path, 'result_log_grade.txt')
with open(log_name, "a") as log_file:
    log_file.write('================ Test Results ground truth ================\n')
    
with open(log_name, "a") as log_file:
    log_file.write('%s\n' % msg)  # save the message


  
  
