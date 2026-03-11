import csv
import argparse
import ast
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

def contain_trigger(trigger, prompt):
    return trigger in prompt


def evaluate_detection(csv_path):
    gt_labels = []      # Ground truth: attack_success
    pred_labels = []    # Predicted: have_trigger
    scores = []         # Pseudo-score: 1.0 if predicted trigger, else 0.0
    time_list = []

    suspicious_counts = []  # size of suspicious_object_dict in each item
    trigger = "kitty" # Define the trigger word to check in prompts

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse boolean flags
            attack_success = row['attack_success'].strip().lower() == 'true'
            prompt = row['prompt'].strip()
            # attack_success = contain_trigger(trigger, prompt)
            have_trigger = row['have_trigger'].strip().lower() == 'true'
            detection_time = float(row['detection_time'].strip())

            gt_labels.append(attack_success)
            pred_labels.append(have_trigger)
            time_list.append(detection_time)

            # Use binary prediction as pseudo-score
            score = 1.0 if have_trigger else 0.0
            scores.append(score)

            # Parse suspicious_object_dict
            try:
                obj_dict = ast.literal_eval(row['suspicious_object_dict'])
                count = len(obj_dict)
            except Exception:
                count = 0
            suspicious_counts.append(count)

    # Compute metrics
    acc = accuracy_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)

    average_detection_time = sum(time_list) / len(time_list)
    avg_suspicious_objects = sum(suspicious_counts) / len(suspicious_counts)

    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).ravel()
    fpr = fp / (fp + tn + 1e-10) 

    try:
        auc = roc_auc_score(gt_labels, scores)
    except:
        auc = -1  # fallback if only one class present

    # Print results
    print("\n📊 Detection Evaluation Metrics")
    print("=" * 40)
    print(f"Total Samples        : {len(gt_labels)}")
    print(f"Attack Success Count : {sum(gt_labels)}")
    print(f"✅ Accuracy           : {acc:.4f}")
    print(f"🎯 Precision          : {precision:.4f}")
    print(f"🔍 Recall (TPR)       : {recall:.4f}")
    print(f"📐 F1 Score           : {f1:.4f}")
    print(f"🚨 False Positive Rate: {fpr:.4f}")
    print(f"📈 AUC (estimated)    : {auc:.4f}")
    print(f"⏱️ Avg Detection Time : {average_detection_time:.4f} seconds")
    print(f"🕵️ Avg Suspicious Objects Detected: {avg_suspicious_objects:.2f}")
    print("=" * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate backdoor detection results from CSV")
    parser.add_argument('--csv', type=str, required=True, help='Path to result CSV file')
    args = parser.parse_args()

    evaluate_detection(args.csv)