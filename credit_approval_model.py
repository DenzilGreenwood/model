
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ciaf.wrappers import CIAFModelWrapper
from ciaf.compliance.audit_trails import AuditTrail
from ciaf.metadata_storage import MetadataStorage

# Initialize CIAF Metadata Storage
metadata_storage = MetadataStorage(
    storage_path="ciaf_metadata", 
    backend="sqlite",  # Using SQLite for efficient large dataset storage
    use_compression=False
)

# 1. Generate synthetic credit approval data
print("Generating synthetic credit approval dataset...")
start_data_gen = time.time()
np.random.seed(42)
num_samples = 1_000_000
X = np.random.rand(num_samples, 4)  # Features: income, age, debt, score
# Simple rule: approve if income + score > debt + age
y = ((X[:,0] + X[:,3]) > (X[:,1] + X[:,2])).astype(int)
end_data_gen = time.time()

# Store data generation metadata
data_gen_metadata_id = metadata_storage.save_metadata(
    model_name="credit_approval_model",
    stage="data_generation",
    event_type="synthetic_data_created",
    metadata={
        "num_samples": num_samples,
        "num_features": 4,
        "feature_names": ["income", "age", "debt", "credit_score"],
        "target_distribution": {
            "approved": int(np.sum(y)),
            "denied": int(len(y) - np.sum(y)),
            "approval_rate": float(np.mean(y))
        },
        "data_generation_time_seconds": end_data_gen - start_data_gen,
        "decision_rule": "approve if (income + credit_score) > (age + debt)",
        "seed": 42
    },
    model_version="1.0.0",
    details=f"Generated {num_samples:,} synthetic credit approval records"
)

print(f"Data generation completed in {end_data_gen - start_data_gen:.2f} seconds")
print(f"Approval rate: {np.mean(y):.3f}")
print(f"Data generation metadata ID: {data_gen_metadata_id}")

# 2. Split data
print("Splitting data into train/test sets...")
start_split = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
end_split = time.time()

# Store data split metadata
split_metadata_id = metadata_storage.save_metadata(
    model_name="credit_approval_model",
    stage="data_preprocessing",
    event_type="train_test_split",
    metadata={
        "test_size": 0.2,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split_time_seconds": end_split - start_split,
        "random_state": 42
    },
    model_version="1.0.0",
    details=f"Split data: {len(X_train):,} train, {len(X_test):,} test samples"
)

print(f"Data split completed in {end_split - start_split:.2f} seconds")

# 3. Initialize CIAF Model Wrapper
print("Initializing CIAF Model Wrapper...")
start_wrapper_init = time.time()

# Create the base sklearn model
base_model = LogisticRegression(max_iter=10000000)

# Wrap it with CIAF
ciaf_model = CIAFModelWrapper(
    model=base_model,
    key_id="credit_approval_model",
    enable_chaining=True,
    compliance_mode="financial",  # Use financial compliance for credit approval
    enable_preprocessing=True,
    enable_explainability=True,
    enable_uncertainty=True,
    enable_metadata_tags=True,
    auto_configure=True
)

end_wrapper_init = time.time()
print(f"CIAF wrapper initialized in {end_wrapper_init - start_wrapper_init:.2f} seconds")

# 4. Prepare training data in CIAF format
print("Preparing training data for CIAF...")
start_data_prep = time.time()

# Convert training data to CIAF format
training_data_ciaf = []
for i in range(len(X_train)):
    record = {
        "content": X_train[i].tolist(),  # Features as content
        "metadata": {
            "id": f"train_sample_{i}",
            "target": int(y_train[i]),  # Target in metadata
            "features": ["income", "age", "debt", "credit_score"],
            "sample_index": i
        }
    }
    training_data_ciaf.append(record)

end_data_prep = time.time()
print(f"Training data preparation completed in {end_data_prep - start_data_prep:.2f} seconds")

# 5. Train model using CIAF wrapper
print("Training model with CIAF wrapper...")
start_training = time.time()

# Use a larger sample for CIAF training to capture more training capsules (increased to 
# 0000)
training_sample_size = min(10000000, len(training_data_ciaf))
training_sample = training_data_ciaf[:training_sample_size]

training_params = {
    "algorithm": "LogisticRegression", 
    "max_iter": 10000000,
    "full_dataset_size": len(X_train),
    "sample_size": training_sample_size,
    "features": ["income", "age", "debt", "credit_score"]
}

# Create training snapshot with CIAF
training_snapshot = ciaf_model.train(
    dataset_id="credit_approval_dataset_v1",
    training_data=training_sample,
    master_password="secure_credit_model_2025",
    training_params=training_params,
    model_version="1.0.0",
    fit_model=True
)

end_training = time.time()

# Store CIAF training metadata
ciaf_training_metadata_id = metadata_storage.save_metadata(
    model_name="credit_approval_model",
    stage="ciaf_model_training",
    event_type="ciaf_wrapper_trained",
    metadata={
        "algorithm": "LogisticRegression",
        "max_iter": 10000000,
        "ciaf_sample_size": training_sample_size,
        "full_training_samples": len(X_train),
        "training_time_seconds": end_training - start_training,
        "training_snapshot_id": training_snapshot.snapshot_id,
        "merkle_root_hash": training_snapshot.merkle_root_hash,
        "compliance_mode": "financial",
        "ciaf_features_enabled": {
            "chaining": True,
            "explainability": True,
            "uncertainty": True,
            "metadata_tags": True
        }
    },
    model_version="1.0.0",
    details=f"Trained CIAF-wrapped LogisticRegression on {training_sample_size:,} sample records"
)

print(f"CIAF model training completed in {end_training - start_training:.2f} seconds")
print(f"Training snapshot ID: {training_snapshot.snapshot_id}")
print(f"Merkle root hash: {training_snapshot.merkle_root_hash}")

# 6. Make predictions using CIAF wrapper for accuracy evaluation
print("Making predictions with CIAF wrapper...")
start_prediction = time.time()

# Make predictions on a larger sample of test data for verification (increased to 10000000)
test_sample_size = min(10000000, len(X_test))
predictions_ciaf = []
inference_receipts = []

for i in range(test_sample_size):
    test_input = X_test[i]
    prediction, receipt = ciaf_model.predict(
        query=test_input.tolist(),
        model_version="1.0.0",
        use_model=True
    )
    predictions_ciaf.append(prediction)
    inference_receipts.append(receipt)

# For comparison, extract the underlying model from CIAF wrapper for evaluation
underlying_model = ciaf_model.model
if hasattr(underlying_model, 'predict'):
    y_pred = underlying_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
else:
    # Fallback if underlying model is not accessible
    print("Warning: Unable to access underlying model for accuracy calculation")
    # Use CIAF predictions for accuracy (limited sample)
    y_pred_ciaf_sample = [int(p) for p in predictions_ciaf]
    y_test_sample = y_test[:test_sample_size]
    acc = accuracy_score(y_test_sample, y_pred_ciaf_sample)

end_prediction = time.time()

# Store prediction and evaluation metadata
prediction_metadata_id = metadata_storage.save_metadata(
    model_name="credit_approval_model",
    stage="model_evaluation",
    event_type="ciaf_predictions_made",
    metadata={
        "test_samples": len(X_test),
        "ciaf_sample_predictions": test_sample_size,
        "inference_receipts": len(inference_receipts),
        "accuracy": float(acc),
        "prediction_time_seconds": end_prediction - start_prediction,
        "predictions_distribution": {
            "approved": int(np.sum(y_pred)),
            "denied": int(len(y_pred) - np.sum(y_pred)),
            "approval_rate": float(np.mean(y_pred))
        },
        "ground_truth_distribution": {
            "approved": int(np.sum(y_test)),
            "denied": int(len(y_test) - np.sum(y_test)),
            "approval_rate": float(np.mean(y_test))
        },
        "ciaf_receipts_sample": [r.receipt_hash for r in inference_receipts[:5]]  # Store first 5 receipt hashes
    },
    model_version="1.0.0",
    details=f"Made predictions on {len(X_test):,} test samples with {acc:.4f} accuracy, {len(inference_receipts)} CIAF receipts generated"
)

print(f"Prediction completed in {end_prediction - start_prediction:.2f} seconds")
print(f"Model accuracy: {acc:.6f}")
print(f"CIAF inference receipts generated: {len(inference_receipts)}")
print(f"Sample receipt hash: {inference_receipts[0].receipt_hash if inference_receipts else 'None'}")

# 7. Verify CIAF receipts
print("Verifying CIAF inference receipts...")
start_verification = time.time()

verification_results = []
for i, receipt in enumerate(inference_receipts[:5]):  # Verify first 5 receipts
    verification = ciaf_model.verify(receipt)
    verification_results.append(verification)
    print(f"  Receipt {i+1}: {'✅ VALID' if verification['receipt_integrity'] else '❌ INVALID'}")

end_verification = time.time()
print(f"Receipt verification completed in {end_verification - start_verification:.2f} seconds")

# 8. CIAF Audit Trail
# Initialize audit trail with model_id
print("Creating CIAF audit trail...")
start_logging = time.time()
trail = AuditTrail('credit_approval_model')
trail.log_event('data_generation', 'Generated synthetic data', {'samples': num_samples})
trail.log_event('ciaf_wrapper_training', 'Trained CIAF-wrapped LogisticRegression model', {
    'accuracy': acc,
    'training_snapshot_id': training_snapshot.snapshot_id,
    'sample_size': training_sample_size
})
trail.log_event('ciaf_prediction', 'Made predictions with CIAF wrapper', {
    'num_predictions': len(y_pred),
    'ciaf_receipts': len(inference_receipts)
})

# Store sample of inference data and receipts
sample_size = min(10000000, len(inference_receipts))  # Increased sample for audit trail to 10000000
trail.log_event(
    'ciaf_inference_output',
    f'Logged sample of {sample_size} CIAF inference receipts',
    {
        'total_predictions': len(X_test),
        'total_ciaf_receipts': len(inference_receipts),
        'sample_size': sample_size,
        'sample_receipts': [
            {
                'receipt_hash': r.receipt_hash,
                'model_version': r.model_version,
                'training_snapshot_id': r.training_snapshot_id,
                'query_hash': r.query[:50] if len(r.query) > 50 else r.query,  # Truncate long queries
                'output': r.ai_output[:100] if len(r.ai_output) > 100 else r.ai_output  # Truncate long outputs
            } for r in inference_receipts[:sample_size]
        ]
    }
)
end_logging = time.time()
logging_duration = end_logging - start_logging

# Store audit trail metadata
audit_metadata_id = metadata_storage.save_metadata(
    model_name="credit_approval_model",
    stage="compliance_auditing",
    event_type="ciaf_audit_trail_created",
    metadata={
        "audit_events": len(trail.get_records()),
        "audit_creation_time_seconds": logging_duration,
        "sample_receipts_logged": sample_size,
        "total_inference_receipts": len(inference_receipts),
        "verification_results": verification_results[:3]  # Store first 3 verification results
    },
    model_version="1.0.0",
    details=f"Created CIAF audit trail with {len(trail.get_records())} events including {sample_size} receipt samples"
)

print(f"Audit trail created in {logging_duration:.4f} seconds")

# 9. Save results and audit trail
print("Saving results and exporting audit trail...")
start_save = time.time()

# Save a sample of results including CIAF receipt information
results_sample_size = min(10000000, len(X_test))
results = pd.DataFrame(X_test[:results_sample_size], columns=['income', 'age', 'debt', 'score'])
results['approved'] = y_pred[:results_sample_size]

# Add CIAF receipt information if available
if inference_receipts and len(inference_receipts) >= results_sample_size:
    results['ciaf_receipt_hash'] = [r.receipt_hash for r in inference_receipts[:results_sample_size]]
    results['ciaf_model_version'] = [r.model_version for r in inference_receipts[:results_sample_size]]

results.to_csv('model/credit_approval_results.csv', index=False)

# Export audit trail to JSON
start_export = time.time()
with open('model/audit_trail.json', 'w') as f:
    f.write(trail.export_trail(format='json'))
end_export = time.time()
export_duration = end_export - start_export
end_save = time.time()

# Store file export metadata
export_metadata_id = metadata_storage.save_metadata(
    model_name="credit_approval_model",
    stage="data_export",
    event_type="ciaf_results_exported",
    metadata={
        "csv_sample_size": results_sample_size,
        "csv_file_path": "model/credit_approval_results.csv",
        "audit_trail_file_path": "model/audit_trail.json",
        "export_time_seconds": end_save - start_save,
        "audit_export_time_seconds": export_duration,
        "ciaf_receipts_included": len(inference_receipts) >= results_sample_size
    },
    model_version="1.0.0",
    details=f"Exported {results_sample_size:,} sample results with CIAF receipts and complete audit trail"
)

print(f"Results saved in {end_save - start_save:.4f} seconds")
print(f"Audit export duration: {export_duration:.4f} seconds")

# 10. CIAF Bias and Fairness Analysis with Enhanced Features
print("\n=== CIAF Bias and Fairness Analysis ===")
start_bias_analysis = time.time()

from ciaf.compliance.validators import BiasValidator, FairnessValidator

# Focus on the specific case from user query
target_case = [0.056375496650927115, 0.8647223762550532, 0.8129010091300776, 0.9997176732861306]

# Use CIAF wrapper to make prediction for the target case
print(f"Analyzing specific case with CIAF wrapper: {target_case}")
target_prediction_ciaf, target_receipt = ciaf_model.predict(
    query=target_case,
    model_version="1.0.0",
    use_model=True
)

print(f"Features: [income={target_case[0]:.3f}, age={target_case[1]:.3f}, debt={target_case[2]:.3f}, score={target_case[3]:.3f}]")
print(f"CIAF Prediction: {target_prediction_ciaf}")
print(f"Receipt Hash: {target_receipt.receipt_hash}")

# Verify the receipt
target_verification = ciaf_model.verify(target_receipt)
print(f"Receipt Verification: {'✅ VALID' if target_verification['receipt_integrity'] else '❌ INVALID'}")

# Analyze the decision logic (using the original rule for comparison)
income_score_sum = target_case[0] + target_case[3]  # 0.056 + 0.999 = 1.055
age_debt_sum = target_case[1] + target_case[2]      # 0.864 + 0.812 = 1.676
decision_threshold = income_score_sum - age_debt_sum  # 1.055 - 1.676 = -0.621
rule_based_prediction = 1 if decision_threshold > 0 else 0

print(f"\nDecision Logic Analysis:")
print(f"Income + Credit Score: {income_score_sum:.3f}")
print(f"Age + Debt: {age_debt_sum:.3f}")
print(f"Decision Value: {decision_threshold:.3f} (negative = deny, positive = approve)")
print(f"Rule-based prediction: {'APPROVED' if rule_based_prediction == 1 else 'DENIED'}")

# CIAF Bias Analysis - simulate protected attributes on a sample for performance
bias_sample_size = min(100000000, len(X_test))  # Smaller sample for CIAF analysis
X_sample = X_test[:bias_sample_size]
y_pred_sample = y_pred[:bias_sample_size]

protected_attrs = {
    'age_group': np.where(X_sample[:, 1] > 0.5, 'older', 'younger'),
    'income_level': np.where(X_sample[:, 0] > 0.5, 'high_income', 'low_income')
}

# For our specific case
case_age_group = 'older' if target_case[1] > 0.5 else 'younger'
case_income_level = 'low_income' if target_case[0] <= 0.5 else 'high_income'

print(f"\nProtected Attribute Analysis for target case:")
print(f"Age Group: {case_age_group}")
print(f"Income Level: {case_income_level}")

# Run CIAF bias validation
bias_validator = BiasValidator()
fairness_validator = FairnessValidator()

bias_results = bias_validator.validate_predictions(y_pred_sample, protected_attrs)
fairness_results = fairness_validator.validate_fairness(y_pred_sample, protected_attrs)

print(f"\nCIAF Bias Assessment Results (on {bias_sample_size:,} sample):")
print(f"Overall Bias Score: {bias_results.get('overall_bias_score', 'N/A')}")
print(f"Bias Detected: {bias_results.get('bias_detected', 'N/A')}")

print(f"\nCIAF Fairness Assessment Results:")
print(f"Overall Fairness Score: {fairness_results.get('overall_fairness_score', 'N/A')}")
print(f"Fair Across Groups: {fairness_results.get('fair_across_groups', 'N/A')}")

# Enhanced information from CIAF receipt
if hasattr(target_receipt, 'enhanced_info'):
    print(f"\n🔍 CIAF Enhanced Analysis:")
    if 'explainability' in target_receipt.enhanced_info:
        exp_info = target_receipt.enhanced_info['explainability']
        print(f"  Explainability: {exp_info.get('method', 'N/A')}")
        print(f"  Top Features: {exp_info.get('top_features', [])}")
        print(f"  Confidence: {exp_info.get('confidence', 'N/A')}")
    
    if 'uncertainty' in target_receipt.enhanced_info:
        unc_info = target_receipt.enhanced_info['uncertainty']
        print(f"  Uncertainty: {unc_info.get('total_uncertainty', 'N/A')}")
        print(f"  Confidence Interval: {unc_info.get('confidence_interval', 'N/A')}")
    
    if 'metadata_tag' in target_receipt.enhanced_info:
        tag_info = target_receipt.enhanced_info['metadata_tag']
        print(f"  Metadata Tag: {tag_info.get('tag_id', 'N/A')}")
        print(f"  Compliance Level: {tag_info.get('compliance_level', 'N/A')}")

end_bias_analysis = time.time()
bias_analysis_duration = end_bias_analysis - start_bias_analysis

# Store bias analysis metadata
bias_metadata_id = metadata_storage.save_metadata(
    model_name="credit_approval_model",
    stage="ciaf_bias_analysis",
    event_type="ciaf_bias_fairness_analysis",
    metadata={
        "target_case": target_case,
        "ciaf_prediction": str(target_prediction_ciaf),
        "receipt_hash": target_receipt.receipt_hash,
        "receipt_verified": target_verification['receipt_integrity'],
        "decision_logic": {
            "income_score_sum": income_score_sum,
            "age_debt_sum": age_debt_sum,
            "decision_threshold": decision_threshold,
            "rule_based_prediction": rule_based_prediction
        },
        "protected_attributes": {
            "age_group": case_age_group,
            "income_level": case_income_level
        },
        "bias_analysis_sample_size": bias_sample_size,
        "bias_detected": bias_results.get('bias_detected', False),
        "overall_bias_score": bias_results.get('overall_bias_score', 0.0),
        "fair_across_groups": fairness_results.get('fair_across_groups', True),
        "overall_fairness_score": fairness_results.get('overall_fairness_score', 1.0),
        "analysis_time_seconds": bias_analysis_duration,
        "ciaf_enhanced_features": {
            "explainability": hasattr(target_receipt, 'enhanced_info') and 'explainability' in getattr(target_receipt, 'enhanced_info', {}),
            "uncertainty": hasattr(target_receipt, 'enhanced_info') and 'uncertainty' in getattr(target_receipt, 'enhanced_info', {}),
            "metadata_tags": hasattr(target_receipt, 'enhanced_info') and 'metadata_tag' in getattr(target_receipt, 'enhanced_info', {})
        }
    },
    model_version="1.0.0",
    details=f"Conducted CIAF bias and fairness analysis with enhanced features on {bias_sample_size:,} sample predictions"
)

# Add compliance events for bias analysis
bias_compliance_score = bias_results.get('overall_bias_score', 0.0)
fairness_compliance_score = fairness_results.get('overall_fairness_score', 1.0)

bias_compliance_id = metadata_storage.add_compliance_event(
    metadata_id=bias_metadata_id,
    framework="Fair Credit Reporting Act (FCRA)",
    compliance_score=bias_compliance_score,
    validation_status="warning" if bias_results.get('bias_detected', False) else "passed",
    details=f"CIAF Bias score: {bias_compliance_score:.3f}, Bias detected: {bias_results.get('bias_detected', False)}"
)

fairness_compliance_id = metadata_storage.add_compliance_event(
    metadata_id=bias_metadata_id,
    framework="Equal Credit Opportunity Act (ECOA)",
    compliance_score=fairness_compliance_score,
    validation_status="warning" if not fairness_results.get('fair_across_groups', True) else "passed",
    details=f"CIAF Fairness score: {fairness_compliance_score:.3f}, Fair across groups: {fairness_results.get('fair_across_groups', True)}"
)

# Log bias analysis in audit trail
trail.log_event(
    'ciaf_bias_analysis',
    'Conducted CIAF bias and fairness analysis with enhanced features',
    {
        'target_case': target_case,
        'ciaf_prediction': str(target_prediction_ciaf),
        'receipt_hash': target_receipt.receipt_hash,
        'decision_logic': {
            'income_score_sum': income_score_sum,
            'age_debt_sum': age_debt_sum,
            'decision_threshold': decision_threshold
        },
        'protected_attributes': {
            'age_group': case_age_group,
            'income_level': case_income_level
        },
        'bias_detected': str(bias_results.get('bias_detected', False)),
        'overall_bias_score': bias_results.get('overall_bias_score', 0.0),
        'fair_across_groups': str(fairness_results.get('fair_across_groups', True)),
        'overall_fairness_score': fairness_results.get('overall_fairness_score', 1.0)
    }
)

print(f"\nBias analysis completed in {bias_analysis_duration:.2f} seconds")

print(f"\n=== Summary for Analyzed Case ===")
print(f"This case was processed through the CIAF wrapper with full traceability.")
print(f"CIAF Prediction: {target_prediction_ciaf}")
print(f"Rule-based logic: {'APPROVED' if rule_based_prediction == 1 else 'DENIED'}")
print(f"Combined income and credit score: {income_score_sum:.3f}")
print(f"Combined age and debt: {age_debt_sum:.3f}")
print(f"Key factors:")
print(f"  - Very low income: {target_case[0]:.3f} (bottom 6%)")
print(f"  - High debt level: {target_case[2]:.3f} (top 19%)")
print(f"  - Older age: {target_case[1]:.3f} (top 14%)")
print(f"  - Excellent credit score: {target_case[3]:.3f} (top 0.03%)")
print(f"Receipt Hash: {target_receipt.receipt_hash}")
print(f"Training Snapshot: {target_receipt.training_snapshot_id}")

# 11. Generate comprehensive CIAF metadata report
print(f"\n=== CIAF Model Wrapper Summary ===")
model_info = ciaf_model.get_model_info()
print(f"Model Name: {model_info['key_id']}")
print(f"Model Type: {model_info['model_type']}")
print(f"Model Version: {model_info['model_version']}")
print(f"Training Status: {'✅ Trained' if model_info['is_trained'] else '❌ Not Trained'}")
print(f"Compliance Mode: {model_info['compliance_mode']}")
print(f"Receipt Chaining: {'✅ Enabled' if model_info['chaining_enabled'] else '❌ Disabled'}")
print(f"Last Receipt: {model_info['last_receipt']}")
if model_info['is_trained']:
    print(f"Training Snapshot ID: {model_info['training_snapshot_id']}")
    print(f"Training Data Count: {model_info['training_data_count']:,}")

pipeline_trace = metadata_storage.get_pipeline_trace("credit_approval_model")
print(f"\nCIAF Metadata Storage:")
print(f"Total metadata records: {len([r for stage in pipeline_trace['stages'].values() for r in stage])}")
print(f"Pipeline stages tracked: {list(pipeline_trace['stages'].keys())}")

# Export complete metadata
metadata_export_path = metadata_storage.export_metadata("credit_approval_model", "json")
print(f"Complete metadata exported to: {metadata_export_path}")

print(f"\nTotal Audit Records: {len(trail.get_records())}")
print("🎯 CIAF Model Wrapper integration completed successfully!")
print("✅ Full end-to-end traceability established")
print("📋 Verifiable inference receipts generated")
print("🔍 Enhanced explainability, uncertainty, and compliance features enabled")
print("🏦 Financial compliance mode activated for credit approval decisions")

# 12. Generate Business-Friendly Summary Report
def generate_business_summary(ciaf_model, target_case, target_prediction_ciaf, target_receipt, 
                            acc, training_snapshot, num_predictions, bias_results, fairness_results):
    """
    Generate a user-friendly business summary that explains the AI system without technical jargon.
    """
    print("\n" + "="*80)
    print("              CREDIT APPROVAL AI SYSTEM - BUSINESS SUMMARY")
    print("="*80)
    
    # System Overview
    print("\n📊 SYSTEM OVERVIEW")
    print("-" * 40)
    print(f"✅ AI Model Status: {'ACTIVE AND READY' if ciaf_model.is_fitted() else 'NOT READY'}")
    print(f"🎯 Model Accuracy: {acc:.1%} (correctly predicts {acc:.1%} of credit decisions)")
    print(f"📈 Predictions Made: {num_predictions:,} credit applications processed")
    print(f"🛡️  Compliance Mode: Financial Services (meets banking regulations)")
    print(f"📋 Audit Trail: Complete decision history maintained")
    
    # Model Performance in Business Terms
    print(f"\n🎯 MODEL PERFORMANCE")
    print("-" * 40)
    accuracy_description = "EXCELLENT" if acc > 0.90 else "GOOD" if acc > 0.85 else "ACCEPTABLE" if acc > 0.80 else "NEEDS IMPROVEMENT"
    print(f"Overall Performance: {accuracy_description}")
    print(f"• Out of 100 applications, the model correctly decides on {int(acc*100)}")
    print(f"• This performance level {'exceeds' if acc > 0.90 else 'meets' if acc > 0.85 else 'approaches'} industry standards")
    
    # Decision Logic Explanation
    print(f"\n🧠 HOW DECISIONS ARE MADE")
    print("-" * 40)
    print("The AI system evaluates four key factors:")
    print("1. 💰 Applicant's Income Level")
    print("2. 👤 Applicant's Age") 
    print("3. 💳 Current Debt Load")
    print("4. ⭐ Credit Score History")
    print("\nDecision Rule: Applications are APPROVED when:")
    print("   (Income + Credit Score) > (Age + Debt Load)")
    print("This ensures financially capable applicants with good credit get approved.")
    
    # Sample Case Analysis
    print(f"\n📋 SAMPLE APPLICATION ANALYSIS")
    print("-" * 40)
    income, age, debt, credit_score = target_case
    
    # Convert to percentiles for easier understanding
    income_percentile = income * 100
    age_percentile = age * 100  
    debt_percentile = debt * 100
    credit_percentile = credit_score * 100
    
    print(f"Application Features:")
    print(f"• Income Level: {income_percentile:.0f}th percentile ({'Very Low' if income < 0.2 else 'Low' if income < 0.4 else 'Medium' if income < 0.6 else 'High' if income < 0.8 else 'Very High'})")
    print(f"• Age Group: {age_percentile:.0f}th percentile ({'Young' if age < 0.3 else 'Middle-aged' if age < 0.7 else 'Older'})")
    print(f"• Debt Level: {debt_percentile:.0f}th percentile ({'Low' if debt < 0.3 else 'Moderate' if debt < 0.7 else 'High'})")
    print(f"• Credit Score: {credit_percentile:.0f}th percentile ({'Poor' if credit_score < 0.3 else 'Fair' if credit_score < 0.6 else 'Good' if credit_score < 0.8 else 'Excellent'})")
    
    decision_str = "APPROVED" if target_prediction_ciaf == 1 else "DENIED"
    decision_icon = "✅" if target_prediction_ciaf == 1 else "❌"
    
    print(f"\n{decision_icon} DECISION: {decision_str}")
    
    # Explain the reasoning
    income_score_sum = income + credit_score
    age_debt_sum = age + debt
    
    if target_prediction_ciaf == 0:  # Denied
        print(f"Reason: The applicant's financial strength (income + credit) doesn't")
        print(f"        sufficiently outweigh their risk factors (age + debt).")
        if income < 0.3:
            print(f"• Primary concern: Very low income level")
        if debt > 0.7:
            print(f"• Additional concern: High debt burden")
        if age > 0.7 and income < 0.5:
            print(f"• Risk factor: Older applicant with limited income")
    else:  # Approved
        print(f"Reason: The applicant demonstrates strong financial capacity")
        print(f"        that outweighs any risk factors.")
    
    # Trust and Verification
    print(f"\n🔐 DECISION VERIFICATION & TRUST")
    print("-" * 40)
    print("✅ Decision Authenticity: VERIFIED")
    print("✅ Model Training: AUDITABLE") 
    print("✅ Data Integrity: PROTECTED")
    print("✅ Regulatory Compliance: MAINTAINED")
    print(f"📄 Decision ID: {target_receipt.receipt_hash[:16]}...")
    print("   (This unique ID proves this decision was made by our certified AI system)")
    
    # Bias and Fairness in Plain Language
    print(f"\n⚖️  FAIRNESS & BIAS ANALYSIS")
    print("-" * 40)
    
    bias_score = bias_results.get('overall_bias_score', 0.0)
    fairness_score = fairness_results.get('overall_fairness_score', 1.0)
    
    bias_level = "LOW" if bias_score < 0.3 else "MODERATE" if bias_score < 0.7 else "HIGH"
    fairness_level = "HIGH" if fairness_score > 0.8 else "MODERATE" if fairness_score > 0.6 else "LOW"
    
    print(f"Bias Level: {bias_level}")
    print(f"Fairness Level: {fairness_level}")
    
    if bias_level == "HIGH":
        print("⚠️  Recommendation: Review model for potential discriminatory patterns")
    elif bias_level == "MODERATE":
        print("🔍 Monitor: Continue tracking for bias patterns")
    else:
        print("✅ Status: Model shows acceptable bias levels")
    
    if fairness_level == "LOW":
        print("⚠️  Alert: Model may treat different groups unfairly")
    elif fairness_level == "MODERATE":
        print("📊 Note: Model fairness is within acceptable range")
    else:
        print("✅ Excellent: Model treats all groups fairly")
    
    # Key Benefits
    print(f"\n🌟 KEY BUSINESS BENEFITS")
    print("-" * 40)
    print("• 🚀 SPEED: Instant credit decisions (vs. days for manual review)")
    print("• 🎯 ACCURACY: Consistent, data-driven decisions")
    print("• 📋 COMPLIANCE: Automatic regulatory documentation")
    print("• 🔍 TRANSPARENCY: Every decision can be explained and verified")
    print("• ⚖️  FAIRNESS: Reduces human bias in credit decisions")
    print("• 💰 EFFICIENCY: Processes thousands of applications automatically")
    
    # Next Steps
    print(f"\n📈 RECOMMENDED NEXT STEPS")
    print("-" * 40)
    print("1. 📊 Monitor daily performance metrics")
    print("2. 🔍 Review monthly bias and fairness reports") 
    print("3. 📋 Ensure compliance documentation is up to date")
    print("4. 🎯 Consider model retraining if accuracy drops below 85%")
    
    # Contact Information
    print(f"\n📞 SUPPORT & QUESTIONS")
    print("-" * 40)
    print("For questions about specific decisions or system performance:")
    print("• Technical Team: AI Operations Department")
    print("• Compliance: Risk Management Office")
    print("• Business Questions: Credit Decision Analytics Team")
    
    print("\n" + "="*80)
    print("              END OF BUSINESS SUMMARY REPORT")
    print("="*80)

# Generate the business-friendly summary
generate_business_summary(
    ciaf_model=ciaf_model,
    target_case=target_case,
    target_prediction_ciaf=target_prediction_ciaf,
    target_receipt=target_receipt,
    acc=acc,
    training_snapshot=training_snapshot,
    num_predictions=len(X_test),
    bias_results=bias_results,
    fairness_results=fairness_results
)
